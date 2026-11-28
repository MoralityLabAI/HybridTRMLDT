"""Registered panel, tournament arms A-K, matched-cost controls and the loop-forcing factorial."""
from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode

from .factory3 import Config, Net, count_params
from .gym3 import FAMILIES3, TaskSpec

PANEL = {   # development families: discovery, rungs, calibration
    "proof_resolve": {"train": TaskSpec("proof_resolve", 14, depth=3),
                      "eval": {"dev": TaskSpec("proof_resolve", 14, depth=3), "ood_deep": TaskSpec("proof_resolve", 24, depth=6)}},
    "experiment_portfolio": {"train": TaskSpec("experiment_portfolio", 5),
                             "eval": {"dev": TaskSpec("experiment_portfolio", 5), "ood_long": TaskSpec("experiment_portfolio", 6)}},
    "revision_retract": {"train": TaskSpec("revision_retract", 16, distract=2),
                         "eval": {"dev": TaskSpec("revision_retract", 16, distract=2),
                                  "ood_long": TaskSpec("revision_retract", 28, distract=4)}},
    "cex_repair": {"train": TaskSpec("cex_repair", 3, depth=1),
                   "eval": {"dev": TaskSpec("cex_repair", 3, depth=1), "ood_minimal": TaskSpec("cex_repair", 3, depth=0),
                            "ood_more": TaskSpec("cex_repair", 5, depth=4)}},
    "proof_deps": {"train": TaskSpec("proof_deps", 14, depth=3),
                   "eval": {"dev": TaskSpec("proof_deps", 14, depth=3), "ood_deep": TaskSpec("proof_deps", 24, depth=6)}},
}
CONTROL = {   # repaired floor/ceiling controls (local_map sat at 1.000 in calibration; replaced by local_map3)
    "local_map3": {"train": TaskSpec("local_map3", 12), "eval": {"dev": TaskSpec("local_map3", 12), "ood_long": TaskSpec("local_map3", 24)}},
    "cumsum3": {"train": TaskSpec("cumsum3", 8), "eval": {"dev": TaskSpec("cumsum3", 8), "ood_long": TaskSpec("cumsum3", 16)}},
}
FINALS_ONLY = {   # never shown to the proposer, never trained before the freeze
    "research_plan": {"train": TaskSpec("research_plan", 10, depth=3),
                      "eval": {"dev": TaskSpec("research_plan", 10, depth=3), "ood_deep": TaskSpec("research_plan", 16, depth=6)}},
}
ALL = {**PANEL, **CONTROL, **FINALS_ONLY}
PRINCIPAL = ("proof_resolve", "experiment_portfolio", "revision_retract", "cex_repair")

BASE = Config(width=32, heads=2, loops=4, memory="prefix", loss="all", steps=300, batch=32)
HRM = BASE.replace(routing="hrm", n_types=4)

ARMS = {
    "A_single": BASE,                                               # reference: one block, loop forcing
    "B_hrm": HRM,                                                   # MOLT-02 winner (soft HRM mixture)
    "C_mix": BASE.replace(routing="mix", n_types=4),                # no manager
    "D_tied": HRM.replace(tie_modules=True),                        # mixture structure without extra blocks
    "E_param": None,                                                # single block, width matched to B params
    "F_visit": BASE.replace(loops=16),                              # single block, 16 module visits
    "G_flop": None,                                                 # single block, width matched to B FLOPs
    "H_sparse_st": HRM.replace(routing="sparse_st", load_balance=0.01),
    "I_anneal": HRM.replace(routing="anneal"),
    "J_sequential": BASE.replace(routing="sequential", n_types=4),  # composition in series
    "K_distill": BASE.replace(distill="logits_all", distill_coef=1.0),  # single TRM student of B, 4 visits
}
TEACHER = HRM
FACTORIAL = {f"{k}_last": None for k in ("A_single", "B_hrm", "C_mix", "H_sparse_st")}
SEEDS_FINAL = (11, 23, 37, 41, 53, 67, 79, 97)
RUNGS = ({"steps": 300, "seeds": (11,)}, {"steps": 300, "seeds": (11, 23)}, {"steps": 300, "seeds": (11, 23, 37)})
REF_FAMILY = "proof_resolve"


def _measure(cfg: Config, family: str = REF_FAMILY) -> tuple[int, float]:
    fam = FAMILIES3[family]
    torch.manual_seed(0)
    m = Net(cfg, fam.vocab, fam.n_out).eval()
    x = fam.generate(ALL[family]["train"], 8, 1).x
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return count_params(m), fc.get_total_flops() / 8


def match_width(target: float, by: str, base: Config = BASE) -> Config:
    best = None
    for w in range(16, 129, 2):
        c = base.replace(width=w)
        p, f = _measure(c)
        v = p if by == "params" else f
        if best is None or abs(v - target) < best[0]:
            best = (abs(v - target), c)
    return best[1]


def resolved_arms() -> dict[str, Config]:
    arms = dict(ARMS)
    p_b, f_b = _measure(HRM)
    arms["E_param"] = match_width(p_b, "params")
    # At 4 loops the parameter and FLOP matches coincide (width 76), so the FLOP-matched arm
    # spends B's forward FLOPs with 8 loops and a narrower block.
    arms["G_flop"] = match_width(f_b, "flops", base=BASE.replace(loops=8))
    for k in FACTORIAL:
        arms[k] = arms[k[:-5]].replace(loss="last")
    return arms


def accounting() -> dict:
    out = {}
    for name, cfg in resolved_arms().items():
        p, f = _measure(cfg)
        out[name] = {"params": p, "flops_per_example_forward": f, "width": cfg.width, "loops": cfg.loops,
                     "routing": cfg.routing, "n_types": cfg.n_types}
    return out
