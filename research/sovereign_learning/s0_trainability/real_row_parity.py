"""S0 gate step 1: real PTQ1_0 rows, Python codec vs the fork's compiled C dequant.

Usage:
python -m research.sovereign_learning.s0_trainability.real_row_parity \
  --gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --ggml-dll C:/path/to/llama-prism/bin/ggml-base.dll --receipt row_parity.json

The reference is `ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float` from a
PrismML build (`dequantize_row_ptq1_0`), loaded with ctypes. The candidate is
`ptq1_codec.decode_ptq1_block` (trit * FP16 scale). Values must be bit-equal.
Every PTQ1_0 tensor is row-sampled; `--full` tensors are compared in full.
This checks weights only. Activation rotation and forward parity are not tested.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path

from .gguf_reader import GGUFFile, TYPE_NAMES
from .ptq1_codec import GGUF_PTQ1_TYPE, GROUP, PTQ1_BLOCK_BYTES, decode_ptq1_block

DEFAULT_FULL = ("blk.0.attn_gate.weight", "blk.63.ffn_down.weight")


class _Traits(ctypes.Structure):
    # ggml.h struct ggml_type_traits (unchanged between the build and the pin)
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64),
                ("blck_size_interleave", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


_TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)


def load_reference(dll: Path):
    os.add_dll_directory(str(dll.parent))
    lib = ctypes.CDLL(str(dll))
    lib.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    t = lib.ggml_get_type_traits(GGUF_PTQ1_TYPE).contents
    meta = {"type_name": t.type_name.decode(), "blck_size": t.blck_size, "type_size": t.type_size}
    if (t.blck_size, t.type_size) != (GROUP, PTQ1_BLOCK_BYTES) or not t.to_float:
        raise RuntimeError(f"reference traits disagree with the codec contract: {meta}")
    fn = _TO_FLOAT(t.to_float)

    def to_float(raw: bytes, k: int) -> list[float]:
        src = ctypes.create_string_buffer(raw, len(raw))
        dst = (ctypes.c_float * k)()
        fn(src, dst, k)
        return list(dst)
    return lib, to_float, meta


def python_row(raw: bytes) -> list[float]:
    out: list[float] = []
    for pos in range(0, len(raw), PTQ1_BLOCK_BYTES):
        trits, scale = decode_ptq1_block(raw[pos:pos + PTQ1_BLOCK_BYTES])
        f32 = struct.unpack("<f", struct.pack("<f", scale))[0]
        out.extend(struct.unpack(f"<{GROUP}f", struct.pack(f"<{GROUP}f", *(t * f32 for t in trits))))
    return out


def compare_rows(g: GGUFFile, name: str, rows: list[int], to_float) -> dict:
    info = g.tensors[name]
    k = info.row_length
    mismatches, max_abs, nonzero = 0, 0.0, 0
    for r in rows:
        raw = bytes(g.payload(name, first_row=r, rows=1))
        ref, cand = to_float(raw, k), python_row(raw)
        for a, b in zip(ref, cand):
            if a != b and not (a != a and b != b):
                mismatches += 1
                max_abs = max(max_abs, abs(a - b))
            nonzero += a != 0.0
    return {"tensor": name, "dims": list(info.dims), "rows_checked": len(rows),
            "values_checked": len(rows) * k, "mismatches": mismatches,
            "max_abs_diff": max_abs, "nonzero_fraction": nonzero / (len(rows) * k)}


def layout_audit(g: GGUFFile, file_size: int) -> dict:
    spans = sorted((i.offset, i.offset + i.nbytes, i.name) for i in g.tensors.values()
                   if i.nbytes is not None)
    overlaps = [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    unknown = [n for n, i in g.tensors.items() if i.nbytes is None]
    return {"tensors": len(g.tensors), "unknown_layout": unknown, "overlaps": overlaps,
            "data_start": g.data_start, "last_payload_end": spans[-1][1],
            "file_size": file_size, "tail_bytes": file_size - spans[-1][1]}


def rotation_audit(g: GGUFFile) -> dict:
    md = g.metadata
    widths = [int(w) for w in md.get("prism.hadamard.sign_widths", [])]
    values = [int(v) for v in md.get("prism.hadamard.sign_values", [])]
    names = md.get("prism.hadamard.weight_names", [])
    missing = [n for n in names if n not in g.tensors]
    unmatched = sorted({g.tensors[n].row_length for n in names if n in g.tensors} - set(widths))
    return {k.split("prism.hadamard.")[1]: md[k] for k in
            ("prism.hadamard.version", "prism.hadamard.block_size", "prism.hadamard.transform",
             "prism.hadamard.axis", "prism.hadamard.sign_mode", "prism.hadamard.gdn_v_grouped")
            if k in md} | {
        "sign_widths": widths, "sign_values_count": len(values),
        "sign_values_sum_widths_match": sum(widths) == len(values),
        "sign_values_minus_one": values.count(-1),
        "sign_values_only_pm1": set(values) <= {-1, 1},
        "sign_values_sha256": hashlib.sha256(struct.pack(f"<{len(values)}i", *values)).hexdigest(),
        "weight_names": len(names), "weight_names_missing": missing,
        "row_lengths_without_sign_width": unmatched,
        "inverse_weight_names": md.get("prism.hadamard.inverse_weight_names", []),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--ggml-dll", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--rows-per-tensor", type=int, default=4)
    p.add_argument("--full", nargs="*", default=list(DEFAULT_FULL))
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--skip-sha", action="store_true")
    a = p.parse_args()

    t0 = time.time()
    _lib, to_float, ref_meta = load_reference(a.ggml_dll)
    rng = random.Random(a.seed)
    with GGUFFile(a.gguf) as g:
        size = a.gguf.stat().st_size
        ptq = [n for n, i in g.tensors.items() if i.ggml_type == GGUF_PTQ1_TYPE]
        sampled = []
        for n in ptq:
            total = g.tensors[n].rows
            pick = {0, total - 1} | {rng.randrange(total) for _ in range(max(0, a.rows_per_tensor - 2))}
            sampled.append(compare_rows(g, n, sorted(pick), to_float))
        full = [compare_rows(g, n, list(range(g.tensors[n].rows)), to_float) for n in a.full]
        receipt = {
            "schema": "s0_real_row_parity.v1",
            "gguf": {"path": str(a.gguf), "bytes": size,
                     "sha256": None if a.skip_sha else file_sha256(a.gguf),
                     "version": g.version, "alignment": g.alignment,
                     "architecture": g.metadata.get("general.architecture"),
                     "file_type": g.metadata.get("general.file_type")},
            "reference": {"dll": str(a.ggml_dll), "dll_sha256": file_sha256(a.ggml_dll), **ref_meta},
            "candidate": "research/sovereign_learning/s0_trainability/ptq1_codec.py decode_ptq1_block",
            "type_counts": {TYPE_NAMES.get(t, str(t)): sum(i.ggml_type == t for i in g.tensors.values())
                            for t in sorted({i.ggml_type for i in g.tensors.values()})},
            "layout": layout_audit(g, size),
            "rotation_metadata": rotation_audit(g),
            "sampled": {"seed": a.seed, "rows_per_tensor": a.rows_per_tensor, "tensors": len(sampled),
                        "values_checked": sum(s["values_checked"] for s in sampled),
                        "mismatches": sum(s["mismatches"] for s in sampled),
                        "tensors_with_mismatch": [s["tensor"] for s in sampled if s["mismatches"]]},
            "full_tensors": full,
            "not_tested": ["activation rotation / sign application order at runtime",
                           "projection or full-model forward parity", "PQ2_0 cross-format check"],
            "elapsed_s": round(time.time() - t0, 1),
        }
    ok = (receipt["sampled"]["mismatches"] == 0 and all(f["mismatches"] == 0 for f in full)
          and not receipt["layout"]["overlaps"])
    receipt["row_parity"] = "PASS" if ok else "FAIL"
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    summary = {k: receipt[k] for k in ("row_parity", "type_counts", "sampled", "elapsed_s")}
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
