"""Genuine verifiers: Lean 4 subprocess checks, Python unit-test subprocesses, a seeded
Monte Carlo experimental-design environment.

Every verifier returns a receipt (inputs digest, return code, parsed outcome, seconds) so a
task's utilities can be re-derived from receipts and spot-checked by rerunning.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

LEAN_TIMEOUT = 90
PY_TIMEOUT = 10


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- Lean
def find_lean() -> str | None:
    env = os.environ.get("JINN_LEAN")
    if env and Path(env).exists():
        return env
    root = Path.home() / ".elan" / "toolchains"
    if root.exists():
        cands = sorted(root.glob("leanprover--lean4---v4.*/bin/lean*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.parent.parent.name)])
        cands = [c for c in cands if c.name in ("lean", "lean.exe")]
        if cands:
            return str(cands[-1])
    return None


def lean_version(lean: str) -> str:
    return subprocess.run([lean, "--version"], capture_output=True, text=True, encoding="utf-8", timeout=60).stdout.strip()


def run_lean(candidates: dict[str, str], evals: dict[str, str], lean: str, workdir: Path) -> dict:
    """candidates: id -> 'theorem ... := proof' line(s); evals: id -> Bool Lean expression.
    One file; per-candidate success is read from error positions."""
    lines, owner = [], {}
    for cid, text in candidates.items():
        start = len(lines) + 1
        for l in text.splitlines():
            lines.append(l)
        for k in range(start, len(lines) + 1):
            owner[k] = cid
    for cid, expr in evals.items():
        lines.append(f'#eval s!"EVAL_{cid}:{{({expr})}}"')  # expr is a Bool-valued Lean term
    src = "\n".join(lines) + "\n"
    workdir.mkdir(parents=True, exist_ok=True)
    f = workdir / f"t_{sha(src)[:16]}.lean"
    f.write_text(src, encoding="utf-8", newline="\n")
    t0 = time.monotonic()
    try:
        p = subprocess.run([lean, str(f)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=LEAN_TIMEOUT, cwd=str(workdir))
        out, rc, timed_out = p.stdout + p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as exc:
        out, rc, timed_out = (exc.stdout or "") if isinstance(exc.stdout, str) else "", -9, True
    secs = time.monotonic() - t0
    failed = set()
    for m in re.finditer(r":(\d+):\d+: error", out):
        ln = int(m.group(1))
        if ln in owner:
            failed.add(owner[ln])
    for m in re.finditer(r":(\d+):\d+: warning: declaration uses [`']sorry[`']", out):
        ln = int(m.group(1))
        if ln in owner:
            failed.add(owner[ln])
    evals_out = {cid: v == "true" for cid, v in re.findall(r"EVAL_(\w+):(true|false)", out)}
    ok = {cid: (cid not in failed and not timed_out) for cid in candidates}
    try:
        f.unlink()
    except OSError:
        pass
    return {"verifier": "lean", "source_sha256": sha(src), "returncode": rc, "timed_out": timed_out,
            "seconds": round(secs, 3), "proved": ok, "evals": evals_out, "output_sha256": sha(out),
            "output_head": out[:600]}


# --------------------------------------------------------------------------- Python
RUNNER = r'''
import json, math, sys
src = sys.stdin.read()
spec = json.loads(src)
ns = {}
try:
    exec(spec["code"], ns)
    fn = ns[spec["name"]]
except Exception as e:
    print(json.dumps({"passed": 0, "total": len(spec["cases"]), "load_error": type(e).__name__})); sys.exit(0)
passed = 0
first_fail = None
for args, exp in spec["cases"]:
    try:
        got = fn(*args)
        ok = (math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-9) if isinstance(exp, float) and isinstance(got, (int, float)) else got == exp)
    except Exception:
        ok = False
    if ok:
        passed += 1
    elif first_fail is None:
        first_fail = args
print(json.dumps({"passed": passed, "total": len(spec["cases"]), "first_fail": first_fail}))
'''


def run_python(name: str, code: str, cases: list) -> dict:
    payload = json.dumps({"name": name, "code": code, "cases": cases})
    t0 = time.monotonic()
    try:
        p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=payload, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=PY_TIMEOUT)
        res = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"passed": 0, "total": len(cases)}
        rc, to = p.returncode, False
    except subprocess.TimeoutExpired:
        res, rc, to = {"passed": 0, "total": len(cases)}, -9, True
    except (json.JSONDecodeError, IndexError):
        res, rc, to = {"passed": 0, "total": len(cases)}, -1, False
    return {"verifier": "python_unittest", "code_sha256": sha(code), "cases_sha256": sha(json.dumps(cases)),
            "returncode": rc, "timed_out": to, "seconds": round(time.monotonic() - t0, 3),
            "passed": res.get("passed", 0), "total": res.get("total", len(cases)),
            "first_fail": res.get("first_fail"), "load_error": res.get("load_error")}


# --------------------------------------------------------------------------- Monte Carlo design environment
def _z_two_sample(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Welch z statistic per replication (rows)."""
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    return (b.mean(1) - a.mean(1)) / np.sqrt(va / a.shape[1] + vb / b.shape[1] + 1e-12)


def _rank_z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mann-Whitney U normal approximation per replication."""
    n1, n2 = a.shape[1], b.shape[1]
    allv = np.concatenate([a, b], axis=1)
    ranks = allv.argsort(axis=1).argsort(axis=1) + 1.0
    u = ranks[:, n1:].sum(1) - n2 * (n2 + 1) / 2
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return (u - mu) / sd


def simulate_world(world: dict, n: int, rng: np.random.Generator, reps: int) -> tuple[np.ndarray, np.ndarray]:
    """Outcomes (reps, n) for control and treatment under the world's data-generating process."""
    d, kind = world["d"], world["kind"]
    if kind == "normal":
        a = rng.normal(0, 1, (reps, n))
        b = rng.normal(d, 1, (reps, n))
    elif kind == "heavy":  # Student t, 2.5 df, scaled
        a = rng.standard_t(2.5, (reps, n))
        b = rng.standard_t(2.5, (reps, n)) + d
    else:  # clustered binary: clusters of size m share a latent shift
        m, icc = world["cluster_size"], world["icc"]
        k = max(1, n // m)
        def arm(shift):
            u = rng.normal(0, math.sqrt(icc / (1 - icc)), (reps, k)).repeat(m, axis=1)[:, :n]
            if u.shape[1] < n:
                u = np.pad(u, ((0, 0), (0, n - u.shape[1])), mode="edge")
            p = 1 / (1 + np.exp(-(-0.3 + shift + u)))
            return (rng.random((reps, n)) < p).astype(float)
        a, b = arm(0.0), arm(d)
    return a, b


def cluster_means(x: np.ndarray, m: int) -> np.ndarray:
    k = x.shape[1] // m
    return x[:, : k * m].reshape(x.shape[0], k, m).mean(2)


def run_design(world: dict, design: str, seed: int, reps: int = 300) -> dict:
    """Execute a design by simulation; returns P(correct decision) and resources used."""
    rng = np.random.default_rng(seed)
    N, truth = world["n_max"], world["d"] > 0
    zc = 1.959964
    t0 = time.monotonic()

    def decide_effect(a, b, analysis):
        if analysis == "rank":
            z = _rank_z(a, b)
        elif analysis == "cluster":
            m = world.get("cluster_size", 1)
            z = _z_two_sample(cluster_means(a, m), cluster_means(b, m)) if m > 1 else _z_two_sample(a, b)
        else:
            z = _z_two_sample(a, b)
        return z > zc

    def robust_choice():
        return "cluster" if world["kind"] == "clustered" else ("rank" if world["kind"] == "heavy" else "t")

    used = N
    if design == "rct_full":
        a, b = simulate_world(world, N, rng, reps)
        dec = decide_effect(a, b, "t")
    elif design == "robust_rct":
        a, b = simulate_world(world, N, rng, reps)
        dec = decide_effect(a, b, robust_choice())
    elif design == "pilot":
        used = 30
        a, b = simulate_world(world, 30, rng, reps)
        dec = decide_effect(a, b, "t")
    elif design == "sequential":
        a, b = simulate_world(world, N, rng, reps)
        h = N // 2
        z1 = _z_two_sample(a[:, :h], b[:, :h])
        stop = (z1 > 2.797) | (z1 < -0.5)
        z2 = _z_two_sample(a, b)
        dec = np.where(stop, z1 > 2.797, z2 > 1.977)
        used = float(np.where(stop, h, N).mean())
    elif design == "simulate_first":
        # pilot data diagnoses the distribution, then the full study uses the diagnosed analysis
        pa, pb = simulate_world(world, 40, rng, 1)
        pooled = np.concatenate([pa, pb], axis=1)[0]
        kurt = float(((pooled - pooled.mean()) ** 4).mean() / (pooled.var() ** 2 + 1e-12))
        binary = set(np.unique(pooled)) <= {0.0, 1.0}
        analysis = "cluster" if binary and world["kind"] == "clustered" else ("rank" if kurt > 4.5 else "t")
        a, b = simulate_world(world, N, rng, reps)
        dec = decide_effect(a, b, analysis)
        used = N + 40
    elif design == "power_table":
        # explicit power table by simulation at assumed d = 0.3, choose the smallest n with power >= 0.8
        grid = [50, 100, 200, 400, 800]
        chosen = grid[-1]
        for g in grid:
            a, b = simulate_world({**world, "d": 0.3}, g, rng, 120)
            if decide_effect(a, b, "t").mean() >= 0.8:
                chosen = g
                break
        used = min(chosen, 2 * N)
        a, b = simulate_world(world, used, rng, reps)
        dec = decide_effect(a, b, "t")
    elif design == "analytic_n":
        # closed-form n for d = 0.3 at 80% power (normal theory), capped at the budget
        used = min(int(math.ceil(2 * ((zc + 0.8416) / 0.3) ** 2)), N)
        a, b = simulate_world(world, used, rng, reps)
        dec = decide_effect(a, b, "t")
    elif design == "ablation":
        # tests the mechanism component, whose effect is a share of the main effect
        wm = {**world, "d": world["d"] * world["mech_share"]}
        a, b = simulate_world(wm, N // 2, rng, reps)
        dec = decide_effect(a, b, robust_choice())
        used = N // 2
    elif design == "observational":
        a, b = simulate_world(world, N, rng, reps)
        b = b + world["confound"]
        dec = decide_effect(a, b, "t")
    else:
        raise KeyError(design)
    acc = float((dec == truth).mean())
    return {"verifier": "mc_design", "design": design, "seed": seed, "reps": reps, "accuracy": acc,
            "n_used": float(used), "seconds": round(time.monotonic() - t0, 3)}
