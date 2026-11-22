"""HUG-CAL-02 host-independent rank-one activation-control construction.

Requires numpy. The original model and candidate factor tensors must be supplied by
an instrumented worker; this module never loads model weights or performs a GPU run.

A temporal interaction compares one FIXED control matrix in both windows. The
write-rotation family keeps the input-read gate fixed; it is a distinct scientific
comparison from read-rotation controls and should be reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

Q = (0.10, 0.50, 0.90, 0.95)
TOL = np.asarray((0.15, 0.15, 0.15, 0.25), dtype=np.float64)
WINDOWS = ("early", "late")


@dataclass(frozen=True)
class RankOneControl:
    u: np.ndarray
    v: np.ndarray
    family: str
    index: int
    scale: float = 1.0

    def matrix(self) -> np.ndarray:
        return self.scale * np.outer(self.u, self.v)

    def activation_norms(self, x: np.ndarray) -> np.ndarray:
        x = _matrix(x)
        if x.shape[1] != self.v.size:
            raise ValueError("input dimension mismatch")
        return np.abs(x @ self.v) * np.linalg.norm(self.u) * abs(self.scale)


def _vector(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)):
        raise ValueError("vector must be finite, 1-D, and at least length two")
    if np.linalg.norm(a) <= 0:
        raise ValueError("zero vector")
    return a


def _matrix(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError("activation matrix must be finite 2-D with >=2 rows")
    return a


def quantiles(norms: np.ndarray) -> np.ndarray:
    a = np.asarray(norms, dtype=np.float64)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("norms must be a finite nonempty nonnegative vector")
    return np.quantile(a, Q, method="linear")


def matching_report(target: np.ndarray, candidate: np.ndarray) -> dict:
    tq, cq = quantiles(target), quantiles(candidate)
    denom = np.maximum(tq, 1e-12)
    errors = np.abs(cq - tq) / denom
    return {
        "target": tq.tolist(),
        "candidate": cq.tolist(),
        "errors": errors.tolist(),
        "accepted": bool(np.all(errors <= TOL)),
        "max_normalized_error": float(np.max(errors / TOL)),
    }


def window_report(
    target: RankOneControl,
    candidate: RankOneControl,
    x_by_window: Mapping[str, np.ndarray],
) -> dict:
    if set(x_by_window) != set(WINDOWS):
        raise ValueError("must include exactly early and late windows")
    out = {w: matching_report(target.activation_norms(x_by_window[w]),
                              candidate.activation_norms(x_by_window[w]))
           for w in WINDOWS}
    return {
        "accepted_both_windows": all(out[w]["accepted"] for w in WINDOWS),
        "max_normalized_error": max(out[w]["max_normalized_error"] for w in WINDOWS),
        "windows": out,
    }


def rotated_write_controls(u: np.ndarray, v: np.ndarray, count: int = 4, seed: int = 11) -> list[RankOneControl]:
    """Keep v exactly; use output writes orthogonal to original u and one another.

    For any x and any t, ||u' v^T x|| == ||u v^T x|| (up to float error).
    Hence this family's activation matching is exact on unseen data. It does NOT
    test the same null as changing the input gate v.
    """
    u, v = _vector(u), _vector(v)
    if not 1 <= count <= u.size - 1:
        raise ValueError("count must fit orthogonal output dimensions")
    rng = np.random.default_rng(seed)
    basis = [u / np.linalg.norm(u)]
    out = []
    for i in range(count):
        for _ in range(100):
            z = rng.standard_normal(u.size)
            for b in basis:
                z -= float(z @ b) * b
            nz = np.linalg.norm(z)
            if nz > 1e-8:
                break
        else:
            raise RuntimeError("failed to sample independent output direction")
        d = z / nz
        basis.append(d)
        out.append(RankOneControl(d * np.linalg.norm(u), v.copy(), "write_rotation", i))
    return out


class _BatchProjection:
    """Bounded candidate blocks; float64 CUDA only when available and worthwhile."""

    def __init__(self, x: Mapping[str, np.ndarray], backend: str = "auto", block: int = 64):
        if backend not in ("auto", "numpy", "cuda") or block < 1:
            raise ValueError("invalid projection backend or block size")
        self.block = block
        self.torch = None
        if backend != "numpy" and (backend == "cuda" or
                                   sum(a.size for a in x.values()) >= 1_000_000):
            try:
                import torch
                if torch.cuda.is_available():
                    self.torch = torch
            except ImportError:
                pass
        if backend == "cuda" and self.torch is None:
            raise RuntimeError("CUDA torch backend unavailable")
        self.x = (x if self.torch is None else
                  {w: self.torch.as_tensor(np.ascontiguousarray(a), dtype=self.torch.float64,
                                           device="cuda") for w, a in x.items()})

    def batches(self, vectors: np.ndarray, *, quantile_windows: bool = False):
        """Yield (slice, pooled medians, optional per-window quantiles)."""
        for start in range(0, vectors.shape[0], self.block):
            end = min(start + self.block, vectors.shape[0])
            block = np.ascontiguousarray(vectors[start:end].T)
            if self.torch is None:
                projections = [np.abs(self.x[w] @ block) for w in WINDOWS]
                medians = np.median(np.concatenate(projections, axis=0), axis=0)
                qs = ([np.quantile(a, Q, axis=0, method="linear") for a in projections]
                      if quantile_windows else None)
            else:
                t = self.torch
                with t.no_grad():
                    b = t.as_tensor(block, dtype=t.float64, device="cuda")
                    projections = [t.abs(self.x[w] @ b) for w in WINDOWS]
                    # Interpolation matches NumPy's even-length pooled median.
                    medians = t.quantile(t.cat(projections, dim=0), 0.5, dim=0).cpu().numpy()
                    qs = ([t.quantile(a, t.tensor(Q, device="cuda", dtype=t.float64),
                                      dim=0).cpu().numpy() for a in projections]
                          if quantile_windows else None)
            yield slice(start, end), medians, qs


def read_rotation_pool(
    u: np.ndarray,
    v: np.ndarray,
    x_selection: Mapping[str, np.ndarray],
    count: int = 512,
    seed: int = 17,
    max_abs_cosine: float = 0.95,
    ridge_fraction: float = 0.02,
    *, backend: str = "auto", candidate_trace: list[dict] | None = None,
) -> list[RankOneControl]:
    """Diagonal-covariance-aware new input gates, fitted on selection ONLY.

    The pooled variance is an inexpensive whitening proposal distribution. A
    calibrated single amplitude fits pooled-window p50; joint quantile matching
    is checked separately in both windows. Same fixed u',v' is used for both.
    This is candidate generation, not a guarantee that any candidates pass.
    """
    u, v = _vector(u), _vector(v)
    x = {w: _matrix(x_selection[w]) for w in WINDOWS}
    if set(x_selection) != set(WINDOWS) or any(x[w].shape[1] != v.size for w in WINDOWS):
        raise ValueError("bad selection windows or input dimensions")
    if count < 1 or not (0 < max_abs_cosine < 1) or ridge_fraction <= 0:
        raise ValueError("invalid control generator parameters")
    pooled = np.concatenate([x[w] for w in WINDOWS], axis=0)
    diag = pooled.var(axis=0)
    ridge = max(float(diag.mean()) * ridge_fraction, 1e-10)
    inv_std = 1.0 / np.sqrt(diag + ridge)
    v_norm = np.linalg.norm(v)
    unit_v = v / v_norm
    target_median = np.median(np.abs(pooled @ v))
    rng = np.random.default_rng(seed)
    pool = []
    proposed = []
    if candidate_trace is not None:
        candidate_trace.clear()
    for i in range(count):
        record = {"id": i, "scale": None, "cosine": None,
                  "windows": {w: {"target": None, "candidate": None} for w in WINDOWS},
                  "max_normalized_error": None, "selection_passed": False,
                  "selected": False, "rejection_reason": None}
        if candidate_trace is not None:
            candidate_trace.append(record)
        z = rng.standard_normal(v.size) * inv_std
        z -= float(z @ unit_v) * unit_v
        if np.linalg.norm(z) <= 1e-12:
            record["rejection_reason"] = "zero_direction"
            continue
        z /= np.linalg.norm(z)
        # Fixed broad proposal grid, without checking verification outcomes.
        mix = (0.4, 0.7, 1.0)[i % 3]
        direction = (1.0 - mix) * unit_v + mix * z
        direction /= np.linalg.norm(direction)
        cosine = abs(float(direction @ unit_v))
        record["cosine"] = cosine
        if cosine > max_abs_cosine:
            record["rejection_reason"] = "cosine_limit"
        proposed.append((i, v_norm * direction))
    if proposed:
        projector = _BatchProjection(x, backend=backend)
        vectors = np.stack([v_candidate for _, v_candidate in proposed])
        target_q = ({w: quantiles(np.abs(x[w] @ v) * np.linalg.norm(u)) for w in WINDOWS}
                    if candidate_trace is not None else None)
        for sl, medians, qs in projector.batches(vectors, quantile_windows=candidate_trace is not None):
            for j, (i, candidate_v) in enumerate(proposed[sl]):
                candidate_median = medians[j]
                if candidate_median < 1e-12:
                    if candidate_trace is not None:
                        candidate_trace[i]["rejection_reason"] = "zero_median"
                    continue
                scalar = float(target_median / candidate_median)
                if candidate_trace is not None:
                    record = candidate_trace[i]
                    record["scale"] = scalar
                    for wi, w in enumerate(WINDOWS):
                        cq = qs[wi][:, j] * scalar * np.linalg.norm(u)
                        tq = target_q[w]
                        errors = np.abs(cq - tq) / np.maximum(tq, 1e-12)
                        record["windows"][w] = {"target": tq.tolist(), "candidate": cq.tolist(),
                                                "errors": errors.tolist(),
                                                "accepted": bool(np.all(errors <= TOL)),
                                                "max_normalized_error": float(np.max(errors / TOL))}
                    record["max_normalized_error"] = max(record["windows"][w]["max_normalized_error"]
                                                         for w in WINDOWS)
                    record["selection_passed"] = (record["rejection_reason"] is None and
                                                   all(record["windows"][w]["accepted"] for w in WINDOWS))
                if (candidate_trace is None and
                    abs(float(candidate_v @ unit_v)) / v_norm <= max_abs_cosine) or (
                    candidate_trace is not None and candidate_trace[i]["rejection_reason"] is None
                ):
                    pool.append(RankOneControl(u.copy(), candidate_v, "read_rotation", i, scalar))
    return pool


def select_joint_controls(
    target: RankOneControl,
    pool: Sequence[RankOneControl],
    x_selection: Mapping[str, np.ndarray],
    count: int = 4,
    max_pairwise_abs_cosine: float = 0.99,
    *, backend: str = "auto", candidate_trace: list[dict] | None = None,
) -> tuple[list[RankOneControl], list[dict]]:
    """Select four distinct fixed matrices based only on selection distributions."""
    if count < 1:
        raise ValueError("count must be positive")
    if set(x_selection) != set(WINDOWS):
        raise ValueError("must include exactly early and late windows")
    if not pool:
        return [], []
    x = {w: _matrix(x_selection[w]) for w in WINDOWS}
    target_q = {w: quantiles(target.activation_norms(x[w])) for w in WINDOWS}
    projector = _BatchProjection(x, backend=backend)
    vectors = np.stack([c.v * c.scale * np.linalg.norm(c.u) for c in pool])
    scored = []
    for sl, _, qs in projector.batches(vectors, quantile_windows=True):
        for j in range(sl.start, sl.stop):
            windows = {}
            for wi, w in enumerate(WINDOWS):
                tq, cq = target_q[w], qs[wi][:, j - sl.start]
                errors = np.abs(cq - tq) / np.maximum(tq, 1e-12)
                windows[w] = {"target": tq.tolist(), "candidate": cq.tolist(),
                              "errors": errors.tolist(), "accepted": bool(np.all(errors <= TOL)),
                              "max_normalized_error": float(np.max(errors / TOL))}
            r = {"accepted_both_windows": all(windows[w]["accepted"] for w in WINDOWS),
                 "max_normalized_error": max(windows[w]["max_normalized_error"] for w in WINDOWS),
                 "windows": windows}
            c = pool[j]
            if candidate_trace is not None:
                record = candidate_trace[c.index]
                record["windows"] = windows
                record["max_normalized_error"] = r["max_normalized_error"]
                record["selection_passed"] = r["accepted_both_windows"]
                if not r["accepted_both_windows"]:
                    record["rejection_reason"] = "quantile_limit"
            scored.append((r, c))
    scored.sort(key=lambda pair: (not pair[0]["accepted_both_windows"],
                                  pair[0]["max_normalized_error"], pair[1].index))
    chosen, reports = [], []
    for r, candidate in scored:
        if not r["accepted_both_windows"]:
            break
        normalized = candidate.v / np.linalg.norm(candidate.v)
        if any(abs(float(normalized @ (c.v / np.linalg.norm(c.v)))) > max_pairwise_abs_cosine
               and candidate.family == c.family == "read_rotation" for c in chosen):
            if candidate_trace is not None:
                candidate_trace[candidate.index]["rejection_reason"] = "pairwise_cosine_limit"
            continue
        # Export uses the exact reference NumPy report, never an approximate GPU verdict.
        exact = window_report(target, candidate, x)
        if candidate_trace is not None:
            record = candidate_trace[candidate.index]
            record["windows"] = exact["windows"]
            record["max_normalized_error"] = exact["max_normalized_error"]
            record["selection_passed"] = exact["accepted_both_windows"]
        if not exact["accepted_both_windows"]:
            if candidate_trace is not None:
                candidate_trace[candidate.index]["rejection_reason"] = "quantile_limit_cpu"
            continue
        chosen.append(candidate)
        reports.append(exact)
        if candidate_trace is not None:
            candidate_trace[candidate.index]["selected"] = True
        if len(chosen) == count:
            break
    if candidate_trace is not None:
        for record in candidate_trace:
            if record["rejection_reason"] is None and not record["selected"]:
                record["rejection_reason"] = "not_selected"
    return chosen, reports


def verify_joint_controls(
    target: RankOneControl,
    chosen: Sequence[RankOneControl],
    x_verification: Mapping[str, np.ndarray],
    required: int = 4,
) -> dict:
    reports = [window_report(target, c, x_verification) for c in chosen]
    return {
        "accepted": len(chosen) == required and all(r["accepted_both_windows"] for r in reports),
        "n_accepted_joint": sum(r["accepted_both_windows"] for r in reports),
        "n_selected": len(chosen),
        "controls": reports,
    }


def fit_remainder_scalar_selection(
    target_selection: Mapping[str, np.ndarray],
    control_selection: Mapping[str, np.ndarray],
) -> float:
    """Positive scalar fit on selection-only norms pooled over both windows.

    Rescaling a full-rank control changes ALL singular values. Report that
    family separately from exact spectrum-preserving permutations.
    """
    if set(target_selection) != set(WINDOWS) or set(control_selection) != set(WINDOWS):
        raise ValueError("expected both windows")
    tq = np.concatenate([quantiles(target_selection[w]) for w in WINDOWS])
    cq = np.concatenate([quantiles(control_selection[w]) for w in WINDOWS])
    if np.any(cq <= 1e-12):
        raise ValueError("cannot scale a zero-norm candidate")
    # Minimize mean squared LOG quantile mismatch in scale; closed-form optimum.
    return float(np.exp(np.mean(np.log(tq) - np.log(cq))))


def window_effects(clean: np.ndarray, factor: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Return paired control-adjusted log-probability shifts [program,seed,2]."""
    """Paired program-by-seed correct-token log probabilities.

    Inputs: [program,seed,2 windows]; controls: [4,program,seed,2].
    Return [program,seed,2] where each cell is factor-minus-clean minus
    the mean of four corresponding control-minus-clean shifts.
    """
    clean = np.asarray(clean, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if clean.ndim != 3 or clean.shape[-1] != 2 or factor.shape != clean.shape:
        raise ValueError("expected matched [program,seed,2] clean/factor")
    if controls.shape != (4, *clean.shape):
        raise ValueError("expected 4 controls with identical paired shape")
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(factor)) or not np.all(np.isfinite(controls)):
        raise ValueError("non-finite causal outcome")
    adjusted = (factor - clean) - (controls - clean[None]).mean(axis=0)
    return adjusted


def difference_in_differences(clean: np.ndarray, factor: np.ndarray,
                              controls: np.ndarray) -> np.ndarray:
    """Return the late-minus-early paired control-adjusted interaction."""
    adjusted = window_effects(clean, factor, controls)
    return adjusted[:, :, 1] - adjusted[:, :, 0]


def clustered_signflip_p(paired: np.ndarray, seed: int = 91, n_draws: int = 9999) -> float:
    """Two-sided clustered sign-flip test of zero mean program-level interaction.

    Signs apply to source programs, retaining all of their within-program seeds.
    Finite Monte Carlo correction prevents a zero p-value.
    """
    x = np.asarray(paired, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("paired outcomes must be [program,seed]")
    if n_draws < 99:
        raise ValueError("too few permutation draws")
    by_program = x.mean(axis=1)
    observed = abs(float(by_program.mean()))
    rng = np.random.default_rng(seed)
    exceeded = 0
    for i in range(0, n_draws, 1024):
        n = min(1024, n_draws-i)
        signs = rng.integers(0, 2, size=(n, len(by_program))) * 2 - 1
        null_effects = np.abs((signs * by_program).mean(axis=1))
        exceeded += int(np.count_nonzero(null_effects >= observed - 1e-15))
    return (exceeded + 1) / (n_draws + 1)


def clustered_bootstrap_ci(paired: np.ndarray, seed: int = 132, n_boot: int = 2000,
                          confidence: float = 0.95) -> dict:
    """Resample source programs, retain their complete within-program seed panel."""
    x = np.asarray(paired, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("paired outcomes must be [at least 3 programs,seed]")
    if n_boot < 100 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap request")
    rng = np.random.default_rng(seed)
    per_program = x.mean(axis=1)
    indices = rng.integers(0, len(per_program), size=(n_boot, len(per_program)))
    means = per_program[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    return {
        "mean": float(per_program.mean()),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1 - tail)),
        "n_programs": len(per_program),
        "n_seeds": x.shape[1],
        "bootstrap_draws": n_boot,
    }


def holm_rejections(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    if any(not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("invalid p-value")
    sorted_pairs = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    out = {key: False for key in p_values}
    for index, (key, p) in enumerate(sorted_pairs):
        if p > alpha / (len(sorted_pairs) - index):
            break
        out[key] = True
    return out
