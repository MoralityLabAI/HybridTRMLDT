"""Registered development panel, baseline arms and budget rungs (frozen before the campaign)."""
from __future__ import annotations

from .factory import MoltConfig
from .gym import TaskSpec

# Development families used during discovery. Each entry: train spec, dev eval specs.
# The first eval spec is the in-distribution readout; "ood*" specs are shifted.
PANEL = {
    "pointer_chase": {
        "train": TaskSpec("pointer_chase", 6, depth=2, distract=2),
        "eval": {"dev": TaskSpec("pointer_chase", 6, depth=2, distract=2),
                 "ood_depth": TaskSpec("pointer_chase", 6, depth=4, distract=2),
                 "ood_shuffled": TaskSpec("pointer_chase", 6, depth=2, distract=2, variant="shuffled")}},
    "revision": {
        "train": TaskSpec("revision", 16, depth=1, distract=2),
        "eval": {"dev": TaskSpec("revision", 16, depth=1, distract=2),
                 "ood_long": TaskSpec("revision", 28, depth=2, distract=6),
                 "ood_reordered": TaskSpec("revision", 16, depth=1, distract=2, variant="reordered")}},
    "repair": {
        "train": TaskSpec("repair", 8, depth=2, variant="unflagged"),
        "eval": {"dev": TaskSpec("repair", 8, depth=2, variant="unflagged"),
                 "ood_more_errors": TaskSpec("repair", 10, depth=4, variant="unflagged"),
                 "ctl_no_errors": TaskSpec("repair", 8, depth=0, variant="unflagged")}},
    "proof_deps": {
        "train": TaskSpec("proof_deps", 14, depth=3),
        "eval": {"dev": TaskSpec("proof_deps", 14, depth=3),
                 "ood_deep": TaskSpec("proof_deps", 24, depth=6)}},
}
# Controls and withheld families (never shown to the proposer during discovery).
CONTROL = {
    "shallow_copy": {"train": TaskSpec("shallow_copy", 12),
                     "eval": {"dev": TaskSpec("shallow_copy", 12), "ood_long": TaskSpec("shallow_copy", 24)}},
    "cumsum": {"train": TaskSpec("cumsum", 8),
               "eval": {"dev": TaskSpec("cumsum", 8), "ood_long": TaskSpec("cumsum", 16)}},
}
WITHHELD = {
    "bounded_walk": {"train": TaskSpec("bounded_walk", 10),
                     "eval": {"dev": TaskSpec("bounded_walk", 10), "ood_long": TaskSpec("bounded_walk", 20)}},
    "experiment_select": {"train": TaskSpec("experiment_select", 5),
                          "eval": {"dev": TaskSpec("experiment_select", 5),
                                   "ood_long": TaskSpec("experiment_select", 6)}},
}

BASE = MoltConfig(width=32, heads=2, loops=4, steps=300, batch=32)

# Stage-A / matched baseline arms. Names are stable identifiers in every receipt.
BASELINES = {
    "tied": BASE.replace(memory="tied"),
    "prev": BASE.replace(memory="prev"),
    "prefix": BASE.replace(memory="prefix"),
    "prefix_all": BASE.replace(memory="prefix", loss="all"),
    "one_pass": BASE.replace(memory="tied", loops=1),
}

# Successive-halving rungs: (steps, seeds). Candidates and baselines share each rung.
RUNGS = ({"steps": 300, "seeds": (11,)}, {"steps": 300, "seeds": (11, 23)}, {"steps": 300, "seeds": (11, 23, 37)})
FINAL_SEEDS = (11, 23, 37, 41, 53)
