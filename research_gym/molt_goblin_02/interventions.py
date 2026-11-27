"""BPD-style mechanistic interventions on loop updates and loop memory.

Fit a basis for loop updates (h[r+1]-h[r]) on one set of development roots, then on
fresh roots project every loop update onto (a) the top-k fitted subspace, (b) a random
k-dimensional subspace of the same rank (matched physical dose), (c) the fitted
complement. A reusable loop subspace predicts (a) close to clean and (b), (c) far below.
"""
from __future__ import annotations

import torch

from .gym import FAMILIES
from .train import evaluate


@torch.no_grad()
def fit_update_basis(model, spec, n: int, seed: int):
    model.eval()
    batch = FAMILIES[spec.family].generate(spec, n, seed)
    out = model(batch.x, trace=True)
    prev = out["base"]
    deltas = []
    for h in out["states"]:
        deltas.append((h - prev).reshape(-1, h.shape[-1]))
        prev = h
    d = torch.cat(deltas)
    d = d - d.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(d, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    return vh, var


def _projector(basis):
    def proj(delta, r):
        return (delta @ basis.T) @ basis
    return proj


@torch.no_grad()
def subspace_panel(model, spec, fit_seed: int, eval_seed: int, ranks=(2, 4, 8), n: int = 256) -> dict:
    vh, var = fit_update_basis(model, spec, n, fit_seed)
    d = vh.shape[1]
    clean = evaluate(model, spec, n, eval_seed)["exact"]
    g = torch.Generator().manual_seed(eval_seed)
    rows = []
    for k in ranks:
        if k >= d:
            continue
        top = vh[:k]
        q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
        rnd = q.T
        comp = vh[k:]
        rows.append({"rank": k, "explained_var": var[:k].sum().item(),
                     "top_k": evaluate(model, spec, n, eval_seed, project=_projector(top))["exact"],
                     "random_k": evaluate(model, spec, n, eval_seed, project=_projector(rnd))["exact"],
                     "complement": evaluate(model, spec, n, eval_seed, project=_projector(comp))["exact"]})
    return {"spec": spec.key(), "clean": clean, "width": d, "spectrum_top8": var[:8].tolist(), "ranks": rows}
