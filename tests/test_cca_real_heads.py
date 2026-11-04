"""Real v3 checkpoints copied from pinned jinn-or-beast-paper artifacts.

veto_jinn.json: Git blob 0f060ac7ff0b693b73abc80dd1907bfc1cfb22df
constraint_jinn.json: Git blob 2d5950f2120e62c37adea5eb87e5d6bb17daba72
The upstream repo is private; self-contained fixture copies support repeatable CI.
"""

import json
from pathlib import Path

from research_gym.core.constitutional_control import audit_saved_head

ROOT = Path(__file__).parent / "fixtures"
POLICY = frozenset({
    "deliberate_without_final_decision",
    "invent_private_evidence",
    "obey_disconfirmed_authority",
})


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_real_saved_veto_jinn_requires_executing_wrapper():
    record = read("cca_veto_jinn.json")
    raw = audit_saved_head(record, POLICY)
    assert raw["features"] == 13
    assert raw["structural_veto_matches_policy"]
    assert not raw["underlying_linear_reject_certificate"]
    assert not raw["effective_reject_certificate"]  # JSON label is not runtime proof
    assert raw["underlying_witness_tags"]
    verified = audit_saved_head(record, POLICY, wrapper_verified=True)
    assert verified["effective_reject_certificate"]


def test_real_constraint_jinn_only_certifies_nontrain():
    record = read("cca_constraint_jinn.json")
    report = audit_saved_head(record, POLICY)
    assert report["underlying_sufficient_non_train_certificate"]
    assert not report["underlying_linear_reject_certificate"]
    assert not report["structural_veto_matches_policy"]
    assert not report["effective_reject_certificate"]
    assert report["underlying_witness_tags"]


def test_one_pass_projection_of_saved_jinn_heads():
    from fractions import Fraction
    from itertools import product
    from research_gym.core.constitutional_control import (
        LaneHead, StructuralVeto, project_reject_weights,
    )
    for name in ("cca_veto_jinn.json", "cca_constraint_jinn.json"):
        record = read(name)
        head = LaneHead(
            *(tuple(Fraction(str(v)) for v in row) for row in record["weights"]),
            tuple(Fraction(str(v)) for v in record["bias"]),
        )
        indices = frozenset(record["feature_names"].index(tag) for tag in POLICY)
        projected = project_reject_weights(head, indices)
        shield = StructuralVeto(head, indices)
        assert projected.certifies_critical_rejection(indices)
        for bits in product((0, 1), repeat=13):
            assert projected.winner(bits) == shield.winner(bits)
