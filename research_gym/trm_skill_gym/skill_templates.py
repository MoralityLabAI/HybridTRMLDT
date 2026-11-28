"""Text-only SKILL.md templates for the six skill families.

Layout follows DataGoblin's executable skill templates
(training/rlt_gym/hypergame_taste/skills/research_hillclimb/SKILL.md): Trigger,
Objective, Required inputs, Action schema, Procedure, Verifier contract,
Termination.  The Procedure section is the existing skill (arm A) written as
instructions; the text-only Inky arm receives exactly this text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from . import tasks as T

FEATURES: Dict[str, Dict[int, str]] = {
    "repair": {0: "candidate_colour", 1: "conflict_fraction", 2: "degree_over_6", 3: "has_conflict"},
    "retrieval": {0: "is_query", 1: "lexical_overlap_with_query", 10: "out_degree_over_4", 11: "in_degree_over_4"},
    "deduction": {0: "prior_score", 1: "n_dominators_over_3", 2: "n_dominated_over_3", 3: "undominated"},
    "research": {**{i: f"p_outcome1_given_h{i}" for i in range(6)}, 6: "cost", **{7 + i: f"prior_h{i}" for i in range(6)}, 13: "cost_weight"},
    "curation": {**{i: f"x{i}" for i in range(6)}, 6: "label", 7: "position", 8: "identical_earlier_twin"},
    "stopping": {0: "is_stop", 1: "is_reason", 2: "is_retrieve", 3: "is_verify", 4: "cost", 5: "gain", 6: "confidence", 7: "budget_over_6", 8: "reason_decay", 9: "confidence_after_rejection", 10: "already_retrieved"},
}
RELATION: Dict[str, str] = {
    "repair": "rel[i][j]=1: nodes i and j are adjacent and must receive different colours",
    "retrieval": "rel[i][j]=1: item i directly depends on item j",
    "deduction": "rel[i][j]=1: candidate j dominates candidate i",
    "research": "unused",
    "curation": "rel[i][j]=similarity weight between examples i and j (1.0 = identical)",
    "stopping": "unused",
}

TEMPLATES: Dict[str, Dict[str, str]] = {
    "repair": {
        "title": "Repair TRM: repair a rejected candidate using verifier diagnostics",
        "trigger": "A candidate solution (a 2-colouring of a constraint graph) was rejected by the verifier, which reports per-node conflict counts.",
        "objective": "Return the minimum set of nodes whose colour must flip so that every edge joins different colours.",
        "action": '{"selected": [node indices to flip]}',
        "procedure": "1. Flip every node where more than half of its incident edges conflict.\n2. Re-verify; flip any node the verifier still reports with a majority of conflicting edges.\n3. Prefer the smaller of a flip set and its complement.",
    },
    "retrieval": {
        "title": "Retrieval TRM: identify relevant documents, symbols and proof dependencies",
        "trigger": "A query item needs every document/symbol/lemma it transitively depends on.",
        "objective": "Return exactly the transitive dependency set of the query (not the query itself).",
        "action": '{"selected": [item indices]}',
        "procedure": "1. Select the query's direct dependencies.\n2. Add items with high lexical overlap with the query.\n3. After verification, add items the verifier marks as missing frontier dependencies and drop items it marks as orphans.",
    },
    "deduction": {
        "title": "Deduction TRM: guide candidate-lattice reduction",
        "trigger": "A candidate lattice must be reduced: a candidate is eliminated iff some surviving candidate dominates it.",
        "objective": "Return exactly the surviving candidates (the kernel of the dominance relation).",
        "action": '{"selected": [surviving candidate indices]}',
        "procedure": "1. Keep every undominated candidate; eliminate the rest.\n2. After verification, drop survivors that have a surviving dominator and restore eliminated candidates that have no surviving dominator.",
    },
    "research": {
        "title": "Research TRM: rank proposed experiments by expected information gain",
        "trigger": "Several experiments are proposed over a finite hypothesis set with a prior; each has a binary outcome likelihood per hypothesis and a cost.",
        "objective": "Choose the experiment maximising expected information gain (bits) minus cost_weight * cost.",
        "action": '{"selected": [one experiment index]}',
        "procedure": "1. For each experiment compute the prior-weighted variance of its outcome likelihoods.\n2. Score = 8 * variance - cost_weight * cost; choose the maximum.",
    },
    "curation": {
        "title": "Curation TRM: identify inconsistent or low-value training examples",
        "trigger": "A small training set may contain label noise and duplicates.",
        "objective": "Flag every example whose label disagrees with its cluster's majority label, and every exact duplicate of an earlier example.",
        "action": '{"selected": [flagged example indices]}',
        "procedure": "1. Flag an example when its label disagrees with the similarity-weighted vote of its neighbours.\n2. Flag an example that has an identical earlier twin.",
    },
    "stopping": {
        "title": "Stopping TRM: decide whether more reasoning, retrieval or verification is worth its cost",
        "trigger": "An agent holds an answer with a confidence, a remaining step budget and known action costs/gains.",
        "objective": "Choose the first action of the value-maximising policy over the remaining budget (stop pays confidence; reasoning gains decay; retrieval helps once; verification ends the episode on success and resets confidence on failure).",
        "action": '{"selected": [one action index: 0 stop, 1 reason, 2 retrieve, 3 verify]}',
        "procedure": "1. For each action estimate the value of taking it once and then stopping.\n2. Choose the action with the highest one-step value.",
    },
}


def render(skill: str) -> str:
    t = TEMPLATES[skill]
    feats = "\n".join(f"- col {k}: {v}" for k, v in FEATURES[skill].items())
    return f"""# Skill: {t['title']}

## Trigger

{t['trigger']}

## Objective

{t['objective']}

## Required inputs

- Candidate table: up to {T.N} candidates, each with typed features:
{feats}
- Relation: {RELATION[skill]}
- Global observation vector and the verifier feedback channel (verified, ok, diag0, diag1).

## Action schema

Return one JSON object: {t['action']}

## Procedure

{t['procedure']}

## Verifier contract

The host verifier (`research_gym.trm_skill_gym.tasks.verify`, skill `{skill}`) returns a receipt verdict and two
structured diagnostics per candidate. The skill never produces its own receipt.

## Termination

Stop after the visit budget, or earlier when the module's confidence reaches the halting threshold or its
predicted value of another recurrence is not positive.
"""


def write_all(root: Path) -> Dict[str, Path]:
    out = {}
    for skill in T.SKILLS:
        p = Path(root) / skill / "SKILL.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render(skill), encoding="utf-8", newline="\n")
        out[skill] = p
    return out


def serialize_instance(skill: str, inst: T.Instance) -> str:
    """Readable instance for the text-only arm (features rounded to 3 places)."""
    n = int(inst.mask.sum()) if skill != "stopping" else 4
    lines = []
    for i in range(n):
        if inst.mask[i] == 0:
            continue
        f = ", ".join(f"{name}={inst.cand[i, c]:.3f}" for c, name in FEATURES[skill].items())
        lines.append(f"{i}: {f}")
    out = "Candidates:\n" + "\n".join(lines)
    if RELATION[skill] != "unused":
        edges = []
        for i in range(n):
            for j in range(n):
                if inst.rel[i, j] > 0:
                    edges.append(f"({i},{j})" if inst.rel[i, j] == 1 else f"({i},{j},{inst.rel[i, j]:.2f})")
        out += f"\nRelation ({RELATION[skill]}):\n" + " ".join(edges)
    return out
