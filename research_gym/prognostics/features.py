"""Architecture features used by the small-data empirical channel."""

from __future__ import annotations

from typing import Any, Mapping


def proposal_features(candidate: Mapping[str, Any]) -> dict[str, float]:
    visits = int(candidate["expanded_visits"])
    physical = int(candidate["physical_modules"])
    gradient = int(candidate["gradient_visible_visits"])
    edges = int(candidate["retained_state_edges"])
    supervision = int(candidate.get("supervision_points", 1))
    tying = str(candidate["tying"])
    return {
        "tied_fraction": {
            "fully_tied": 1.0,
            "grouped": 0.5,
            "alternating": 0.5,
            "untied": 0.0,
        }[tying],
        "physical_module_count": float(physical),
        "expanded_visit_count": float(visits),
        "gradient_visible_fraction": gradient / visits,
        "retained_state_edge_fraction": edges / max(1, visits - 1),
        "supervision_density": supervision / visits,
        "carry_persistent": float(candidate.get("carry") == "persistent"),
        "carry_detached": float(candidate.get("carry") == "detached"),
        "residual_alpha": float(candidate.get("alpha", 1.0)),
        "residual_beta": float(candidate.get("beta", 1.0)),
        "residual_p": float(candidate.get("p", 0.0)),
        "schedule_nesting_depth": float(candidate.get("nesting_depth", 1)),
        "post_norm": float(candidate.get("normalization") == "post"),
    }
