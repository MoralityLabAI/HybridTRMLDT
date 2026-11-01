"""S0 gate step 2a: one real projection, fork CPU runtime vs the PyTorch oracle.

Usage:
python -m research.sovereign_learning.s0_trainability.projection_parity \
  --gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --ggml-bin C:/path/to/llama-prism/bin --receipt projection_parity.json

Builds, with the fork's own ggml-base / ggml-cpu DLLs (ctypes), the graph that
llm_graph_context::build_lora_mm emits for a Hadamard-folded weight:
    h = mul_mat(rot, reshape(x * signs, 1024, ...))  [GGML_HINT_SRC0_IS_HADAMARD]
    y = mul_mat(W, h)
with rot filled exactly as llama-model.cpp does and signs taken from
prism.hadamard.sign_values for the weight's input width. Three fork outputs:
  fork_q    W in PTQ1_0 (runtime path: activations quantized to Q8_0)
  fork_f32  W dequantized to F32 by the fork's to_float (no activation quantization)
  h_dense   the rotation as a plain F32 matmul, without the FWHT hint
The oracle is packed_linear.FrozenTernaryLinear in float64.
Inputs are seeded Gaussian vectors, not real model activations.

Pass criteria (fixed before the first run):
  A. FWHT kernel h vs oracle transformed input: max |diff| / rms(oracle) < 1e-5
  B. fork_f32 vs oracle: max |diff| / rms(oracle) < 1e-4
  C. rms(fork_q - oracle) <= 1.05 * rms(fork_q - fork_f32): any fork_q error is
     the fork's own activation quantization, not a weight/sign/transform mismatch
The ratio C and rms(fork_q - fork_f32) are the measured quantization spread that
a later full-logit tolerance can be based on.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import struct
import time
from pathlib import Path

import torch

from .gguf_reader import GGUFFile
from .packed_linear import FrozenTernaryLinear
from .ptq1_codec import GGUF_PTQ1_TYPE

GGML_TYPE_F32 = 0
GGML_HINT_SRC0_IS_HADAMARD = 1
DEFAULT_TENSOR = "blk.0.attn_gate.weight"
vp, i64 = ctypes.c_void_p, ctypes.c_int64


class _InitParams(ctypes.Structure):
    _fields_ = [("mem_size", ctypes.c_size_t), ("mem_buffer", vp), ("no_alloc", ctypes.c_bool)]


class _Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", i64), ("blck_size_interleave", i64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool),
                ("to_float", vp), ("from_float_ref", vp)]


class Ggml:
    def __init__(self, bin_dir: Path, cpu_variant: str):
        os.add_dll_directory(str(bin_dir))
        b = ctypes.CDLL(str(bin_dir / "ggml-base.dll"))
        c = ctypes.CDLL(str(bin_dir / f"ggml-cpu-{cpu_variant}.dll"))
        sig = {"ggml_init": (vp, [_InitParams]), "ggml_free": (None, [vp]),
               "ggml_new_tensor_1d": (vp, [vp, ctypes.c_int, i64]),
               "ggml_new_tensor_2d": (vp, [vp, ctypes.c_int, i64, i64]),
               "ggml_mul": (vp, [vp, vp, vp]), "ggml_mul_mat": (vp, [vp, vp, vp]),
               "ggml_mul_mat_set_hint": (None, [vp, ctypes.c_int]),
               "ggml_reshape_2d": (vp, [vp, vp, i64, i64]),
               "ggml_reshape_4d": (vp, [vp, vp, i64, i64, i64, i64]),
               "ggml_new_graph": (vp, [vp]), "ggml_build_forward_expand": (None, [vp, vp]),
               "ggml_get_data": (vp, [vp]), "ggml_nbytes": (ctypes.c_size_t, [vp]),
               "ggml_get_type_traits": (ctypes.POINTER(_Traits), [ctypes.c_int])}
        for name, (res, args) in sig.items():
            f = getattr(b, name)
            f.restype, f.argtypes = res, args
        c.ggml_graph_compute_with_ctx.restype = ctypes.c_int
        c.ggml_graph_compute_with_ctx.argtypes = [vp, vp, ctypes.c_int]
        c.ggml_cpu_init.restype, c.ggml_cpu_init.argtypes = None, []
        c.ggml_cpu_init()
        self.b, self.c = b, c
        t = b.ggml_get_type_traits(GGUF_PTQ1_TYPE).contents
        self._to_float = ctypes.CFUNCTYPE(None, vp, ctypes.POINTER(ctypes.c_float), i64)(t.to_float)

    def to_float(self, raw: bytes, k: int):
        dst = (ctypes.c_float * k)()
        self._to_float(ctypes.create_string_buffer(raw, len(raw)), dst, k)
        return dst

    def set_data(self, tensor, raw) -> None:
        n = self.b.ggml_nbytes(tensor)
        if len(raw) != n:
            raise ValueError(f"tensor wants {n} bytes, got {len(raw)}")
        ctypes.memmove(self.b.ggml_get_data(tensor), raw, n)

    def read_f32(self, tensor, count: int) -> list[float]:
        return list((ctypes.c_float * count).from_address(self.b.ggml_get_data(tensor)))


def rotation_bytes(n: int) -> bytes:
    # llama-model.cpp: entry (row, col) = +/- 1/sqrt(n), sign = parity(row & col)
    scale = struct.unpack("<f", struct.pack("<f", 1.0 / math.sqrt(n)))[0]
    vals = [(-scale if bin(r & c).count("1") & 1 else scale) for r in range(n) for c in range(n)]
    return struct.pack(f"<{n * n}f", *vals)


def run_fork(g: Ggml, *, w_type: int, w_raw: bytes, k: int, m: int, x: bytes, ntok: int,
             signs: bytes, block: int, hint: bool, threads: int):
    mem = len(w_raw) + block * block * 4 + 6 * k * ntok * 4 + m * ntok * 4 + (64 << 20)
    ctx = g.b.ggml_init(_InitParams(mem, None, False))
    try:
        tw = g.b.ggml_new_tensor_2d(ctx, w_type, k, m)
        tx = g.b.ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ntok)
        ts = g.b.ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k)
        tr = g.b.ggml_new_tensor_2d(ctx, GGML_TYPE_F32, block, block)
        for t, raw in ((tw, w_raw), (tx, x), (ts, signs), (tr, rotation_bytes(block))):
            g.set_data(t, raw)
        xs = g.b.ggml_mul(ctx, tx, ts)
        h = g.b.ggml_mul_mat(ctx, tr, g.b.ggml_reshape_4d(ctx, xs, block, k // block, ntok, 1))
        if hint:
            g.b.ggml_mul_mat_set_hint(h, GGML_HINT_SRC0_IS_HADAMARD)
        h2 = g.b.ggml_reshape_2d(ctx, h, k, ntok)
        y = g.b.ggml_mul_mat(ctx, tw, h2)
        graph = g.b.ggml_new_graph(ctx)
        g.b.ggml_build_forward_expand(graph, y)
        status = g.c.ggml_graph_compute_with_ctx(ctx, graph, threads)
        if status != 0:
            raise RuntimeError(f"ggml_graph_compute_with_ctx status {status}")
        return g.read_f32(h, k * ntok), g.read_f32(y, m * ntok)
    finally:
        g.b.ggml_free(ctx)


def rel_max(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a - ref).abs().max() / ref.pow(2).mean().sqrt())


def rms(t: torch.Tensor) -> float:
    return float(t.pow(2).mean().sqrt())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--ggml-bin", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--tensor", default=DEFAULT_TENSOR)
    p.add_argument("--cpu-variant", default="haswell")
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261101)
    a = p.parse_args()
    t0 = time.time()

    g = Ggml(a.ggml_bin, a.cpu_variant)
    with GGUFFile(a.gguf) as f:
        info = f.tensors[a.tensor]
        md = f.metadata
        if info.ggml_type != GGUF_PTQ1_TYPE or a.tensor not in md["prism.hadamard.weight_names"]:
            raise SystemExit(f"{a.tensor} is not a Hadamard-folded PTQ1_0 weight")
        if ".ssm_out." in a.tensor:
            raise SystemExit("ssm_out carries the GDN feature permutation, which this script does not build")
        k, m, block = info.row_length, info.rows, int(md["prism.hadamard.block_size"])
        widths, values = md["prism.hadamard.sign_widths"], md["prism.hadamard.sign_values"]
        off = sum(widths[:widths.index(k)])
        sign_list = [float(v) for v in values[off:off + k]]
        w_raw = bytes(f.payload(a.tensor))
        gguf_meta = {"path": str(a.gguf), "sha256": sha_file(a.gguf), "tensor": a.tensor,
                     "dims": list(info.dims), "sign_width_offset": off}

    rng = random.Random(a.seed)
    x_vals = [rng.gauss(0.0, 1.0) for _ in range(k * a.tokens)]
    x = struct.pack(f"<{len(x_vals)}f", *x_vals)
    signs = struct.pack(f"<{k}f", *sign_list)
    w_f32 = bytes(g.to_float(w_raw, k * m))

    common = dict(k=k, m=m, x=x, ntok=a.tokens, signs=signs, block=block, threads=a.threads)
    h_q, y_q = run_fork(g, w_type=GGUF_PTQ1_TYPE, w_raw=w_raw, hint=True, **common)
    _, y_f = run_fork(g, w_type=GGML_TYPE_F32, w_raw=w_f32, hint=True, **common)
    h_dense, _ = run_fork(g, w_type=GGML_TYPE_F32, w_raw=w_f32, hint=False, **common)

    oracle = FrozenTernaryLinear.from_payload(
        w_raw, rows=m, cols=k, signs=torch.tensor(sign_list), rotation_block=block).double()
    xt = torch.tensor(x_vals, dtype=torch.float64).reshape(a.tokens, k)
    with torch.no_grad():
        z_o, y_o = oracle.transformed_input(xt), oracle(xt)
    T = lambda v, n: torch.tensor(v, dtype=torch.float64).reshape(a.tokens, n)
    h_q, h_dense, y_q, y_f = T(h_q, k), T(h_dense, k), T(y_q, m), T(y_f, m)

    crit_a = rel_max(h_q, z_o)
    crit_b = rel_max(y_f, y_o)
    spread = rms(y_q - y_f)
    ratio = rms(y_q - y_o) / spread if spread else float("inf")
    checks = {"A_fwht_vs_oracle_relmax": crit_a, "A_pass": crit_a < 1e-5,
              "B_f32_vs_oracle_relmax": crit_b, "B_pass": crit_b < 1e-4,
              "C_ratio_rms(q-oracle)/rms(q-f32)": ratio, "C_pass": ratio <= 1.05}
    cos = float(torch.nn.functional.cosine_similarity(y_q.flatten(), y_o.flatten(), dim=0))
    receipt = {
        "schema": "s0_projection_parity.v1",
        "gguf": gguf_meta,
        "runtime": {"bin": str(a.ggml_bin), "cpu_variant": a.cpu_variant, "threads": a.threads,
                    "ggml_base_sha256": sha_file(a.ggml_bin / "ggml-base.dll"),
                    "ggml_cpu_sha256": sha_file(a.ggml_bin / f"ggml-cpu-{a.cpu_variant}.dll"),
                    "note": "PrismML build adfffbe41; the pin c96baf4 changed x86 quants.c, ops.cpp and llama-graph.cpp since"},
        "input": {"kind": "seeded Gaussian, not model activations", "seed": a.seed, "tokens": a.tokens},
        "graph": "y = mul_mat(W, reshape(mul_mat_hint(rot, reshape(x*signs, block))))",
        "checks": checks,
        "measured": {"rms_oracle": rms(y_o), "rms(fork_q-fork_f32)": spread,
                     "rms(fork_q-oracle)": rms(y_q - y_o), "maxabs(fork_q-oracle)": float((y_q - y_o).abs().max()),
                     "relmax(fork_q-oracle)": rel_max(y_q, y_o), "cosine(fork_q,oracle)": cos,
                     "fwht_vs_dense_rotation_relmax": rel_max(h_q, h_dense),
                     "dense_rotation_vs_oracle_relmax": rel_max(h_dense, z_o)},
        "not_tested": ["real activations / full forward", "GDN-permuted ssm_out weights",
                       "pin c96baf4 runtime kernels", "CUDA backend", "tokenization"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    ok = checks["A_pass"] and checks["B_pass"] and checks["C_pass"]
    receipt["projection_parity"] = "PASS" if ok else "FAIL"
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("projection_parity", "checks", "measured", "elapsed_s")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
