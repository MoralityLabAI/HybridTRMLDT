"""Tiny dependency-free CPU certificates for symmetric Kronecker sums.

These routines are audit references, not production replacements. The Jacobi
solver is intended only for small symmetric real matrices.
"""
from __future__ import annotations

import math


def _square(a):
    n = len(a)
    if not n or any(len(row) != n for row in a):
        raise ValueError("expected a nonempty square matrix")
    if any(not math.isfinite(x) for row in a for x in row):
        raise ValueError("matrix entries must be finite")
    if any(abs(a[i][j] - a[j][i]) > 1e-10 for i in range(n) for j in range(n)):
        raise ValueError("matrix must be symmetric")
    return n


def eigvalsh_small(matrix, *, tol=1e-13, max_sweeps=100):
    """Symmetric Jacobi rotations; raises if a tiny instance does not converge."""
    n = _square(matrix)
    a = [list(map(float, row)) for row in matrix]
    if n == 1:
        return [a[0][0]]
    for _ in range(max_sweeps * n * n):
        i, j = max(((i, j) for i in range(n) for j in range(i + 1, n)),
                   key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[i][j]) <= tol:
            return sorted(a[k][k] for k in range(n))
        tau = (a[j][j] - a[i][i]) / (2.0 * a[i][j])
        t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1 + tau * tau))
        c = 1 / math.sqrt(1 + t * t)
        s = t * c
        off = a[i][j]
        a[i][i] -= t * off
        a[j][j] += t * off
        a[i][j] = a[j][i] = 0.0
        for k in range(n):
            if k != i and k != j:
                left, right = a[k][i], a[k][j]
                a[k][i] = a[i][k] = c * left - s * right
                a[k][j] = a[j][k] = s * left + c * right
    raise ArithmeticError("Jacobi eigensolver did not converge")


def kronecker_sum(controller, source):
    n, m = _square(controller), _square(source)
    return [[(controller[i // m][j // m] if i % m == j % m else 0.0)
             + (source[i % m][j % m] if i // m == j // m else 0.0)
             for j in range(n * m)] for i in range(n * m)]


def pairwise_spectrum(controller, source):
    return sorted(x + y for x in eigvalsh_small(controller)
                  for y in eigvalsh_small(source))


def normalize_max(eigenvalues):
    """Follow maximum-eigenvalue normalization, with a defined zero case."""
    values = list(eigenvalues)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("empty or nonfinite spectrum")
    if min(values) < -1e-10:
        raise ValueError("requires a PSD spectrum")
    # The production eigensolvers clamp tiny negative roundoff to zero.
    nonnegative = [max(0.0, x) for x in values]
    maximum = max(nonnegative)
    # Production scales whenever maximum > 0, with no near-zero threshold.
    return [x / maximum for x in nonnegative] if maximum > 0.0 else nonnegative


def heat_trace(eigenvalues, time, *, normalized=False):
    if not math.isfinite(time) or time < 0:
        raise ValueError("time must be finite and nonnegative")
    values = normalize_max(eigenvalues) if normalized else list(eigenvalues)
    if not values:
        raise ValueError("empty spectrum")
    return sum(math.exp(-time * x) for x in values) / len(values)


def certificate(controller, source, times=(0.0, 1.0, 4.0, 16.0)):
    """Compare independent explicit-matrix and paired-eigenvalue routes."""
    c, q = eigvalsh_small(controller), eigvalsh_small(source)
    explicit = eigvalsh_small(kronecker_sum(controller, source))
    paired = pairwise_spectrum(controller, source)
    if min(c) < -1e-10 or min(q) < -1e-10:
        raise ValueError("certificate requires PSD matrices")
    tolerance = 1e-9
    if any(abs(x - y) > tolerance for x, y in zip(explicit, paired)):
        raise AssertionError("Kronecker spectral identity failed")
    reports = []
    scale = max(c) + max(q)
    for t in times:
        raw = heat_trace(explicit, t)
        product = heat_trace(c, t) * heat_trace(q, t)
        normal = heat_trace(explicit, t, normalized=True)
        adjusted = (heat_trace(c, t / scale) * heat_trace(q, t / scale)
                    if scale > 0.0 else 1.0)
        if abs(raw - product) > tolerance or abs(normal - adjusted) > tolerance:
            raise AssertionError("heat trace identity failed")
        reports.append({"time": t, "raw": raw, "raw_product": product,
                        "normalized": normal, "scaled_product": adjusted})
    return {"explicit_spectrum": explicit, "pairwise_spectrum": paired,
            "heat": reports}


def production_style_spectrum(controller, source, *, coupling=1.0):
    """Reflect the bridge's factor-then-product maximum normalization.

    Input matrices must be symmetric PSD. Production coupling is frozen at 1.
    This small audit function neither reads nor modifies registered outcomes.
    """
    if not math.isfinite(coupling) or coupling < 0.0:
        raise ValueError("coupling must be finite and nonnegative")
    c = eigvalsh_small(controller)
    q = eigvalsh_small(source)
    c_norm, q_norm = normalize_max(c), normalize_max(q)
    return normalize_max(sorted(x + coupling * y for x in c_norm for y in q_norm))
