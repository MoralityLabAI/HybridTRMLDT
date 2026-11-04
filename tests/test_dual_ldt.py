"""Dual-LDT host-correction, preference, and trust-boundary contracts."""
import json
from itertools import product
from math import exp, isclose, log
from pathlib import Path

import pytest

from research_gym.core.dual_ldt import (
    ActionEvidence, BeastSnapshot, JinnHead, sort_with_jinn, total_variation,
)

ROOT = Path(__file__).parent / "fixtures"
BAN = frozenset({"fabricate_tool_result", "exceed_delegated_scope",
                 "substitute_undeclared_goal", "invent_private_evidence",
                 "assert_known_falsehood"})
ROUTES = {
    "T-a": frozenset({"indirect_route", "staged_test", "pursue_declared_objective", "honor_delegated_scope"}),
    "T-b": frozenset({"indirect_route", "strategic_withholding", "pursue_declared_objective", "honor_delegated_scope"}),
    "P-a": frozenset({"side_with_weaker_party", "public_naming", "pursue_declared_objective", "honor_delegated_scope"}),
    "P-b": frozenset({"side_with_weaker_party", "override_process_for_equity", "pursue_declared_objective", "honor_delegated_scope"}),
    "R-a": frozenset({"seek_consensus", "face_saving_compromise", "pursue_declared_objective", "honor_delegated_scope"}),
    "R-b": frozenset({"seek_consensus", "defer_for_peace", "pursue_declared_objective", "honor_delegated_scope"}),
    "L-a": frozenset({"uniform_rule_application", "explicit_justification", "pursue_declared_objective", "honor_delegated_scope"}),
    "L-b": frozenset({"uniform_rule_application", "explicit_justification", "ignore_circumstance", "pursue_declared_objective", "honor_delegated_scope"}),
    "X": frozenset({"fabricate_tool_result", "pursue_declared_objective"}),
}


def evidence(*, positive=True, negative=False, possible=(frozenset(),)):
    return ActionEvidence(positive, negative, tuple(possible))


def snapshot(*, version="v1", banned=BAN, denied=frozenset(), halted=False,
             tags=ROUTES, unknown=frozenset(), ambiguous=frozenset()):
    observations = {
        a: evidence(positive=True, negative=a in denied,
                    possible=() if a in unknown else
                    ((tagset, frozenset()) if a in ambiguous else (tagset,)))
        for a, tagset in tags.items()
    }
    return BeastSnapshot(version, tuple(tags), frozenset(banned), observations, halted)


def head(name):
    r = json.loads((ROOT / f"dual_jinn_{name}_k0.json").read_text(encoding="utf-8"))
    return JinnHead.from_record(r)


def test_four_evidence_states_and_uncertainty_are_distinct():
    tags = {"a": frozenset(), "b": frozenset({"fabricate_tool_result"}),
            "c": frozenset(), "d": frozenset()}
    s = BeastSnapshot("e1", tuple(tags), BAN, {
        "a": evidence(positive=False, negative=False),
        "b": evidence(possible=(tags["b"],)),
        "c": evidence(positive=True, negative=True),
        "d": evidence(positive=False, negative=True),
    })
    assert [s.status(a) for a in s.actions] == [
        "hold_missing_authority", "reject_forbidden",
        "hold_conflicting_authority", "reject_denied"
    ]
    assert s.authorized() == frozenset()
    assert s.receipt("a") is None


def test_ambiguous_and_unknown_semantics_hold():
    base = snapshot(unknown=frozenset({"T-a"}), ambiguous=frozenset({"T-b"}))
    assert base.status("T-a") == "hold_unknown_semantics"
    # T-b tags plus a safe alternative are both non-critical, so it remains authorized.
    assert base.status("T-b") == "authorized"
    observations = dict(base.evidence)
    observations["T-b"] = evidence(possible=(ROUTES["T-b"], BAN))
    mixed = BeastSnapshot("same", base.actions, BAN, observations)
    assert mixed.status("T-b") == "hold_ambiguous_semantics"
    clean = dict(mixed.evidence)
    clean["T-b"] = evidence(possible=(ROUTES["T-b"],))
    refined = BeastSnapshot("refine", base.actions, BAN, clean)
    assert mixed.authorized() <= refined.authorized()
    assert "T-b" in refined.authorized()


def test_authority_tightening_and_shutdown_revoke_permissions():
    initial = snapshot()
    tightened = snapshot(version="v2", banned=BAN | {"staged_test"})
    assert tightened.authorized() <= initial.authorized()
    assert "T-a" in initial.authorized() and "T-a" not in tightened.authorized()
    paused = snapshot(version="v3", halted=True)
    assert paused.authorized() == frozenset()
    assert all(paused.status(a) == "halted" for a in paused.actions)


def test_stale_or_forged_receipts_are_rejected():
    first = snapshot()
    r = first.receipt("R-a")
    assert r is not None
    assert first.verify_execution(r, lambda epoch, action: True)
    assert not first.verify_execution(r, lambda epoch, action: False)
    assert not snapshot(version="v2").verify_execution(r, lambda *_: True)
    assert not snapshot(halted=True).verify_execution(r, lambda *_: True)
    assert not first.verify_execution(first.receipt("X"), lambda *_: True)
    # Changing the tag interpretation invalidates the digest even at same epoch.
    changed = snapshot(denied=frozenset({"R-a"}))
    assert not changed.verify_execution(r, lambda *_: True)


def test_jinn_cannot_enlarge_beast_authorized_set():
    s = snapshot()
    for name in ("trickster", "partisan", "reconciler", "legalist"):
        h = head(name)
        assert len(h.feature_names) == 37
        for strength in (0., .5, 1., 3.):
            result = sort_with_jinn(s, h, strength=strength)
            assert result.permitted == s.authorized()
            assert "X" not in result.permitted
            assert result.action in s.authorized()
            assert isclose(sum(result.probabilities.values()), 1., abs_tol=1e-12)
            assert isclose(sum(result.original_probabilities.values()), 1., abs_tol=1e-12)
            safe_mass = sum(result.original_probabilities[a] for a in result.permitted)
            assert isclose(result.preference_tax, -log(safe_mass), abs_tol=1e-12)


def test_correction_preserves_relative_odds_when_tags_fixed():
    h = head("partisan")
    first = snapshot()
    later = snapshot(version="second", banned=BAN | {"staged_test"})
    for strength in (0., .1, 1., 4.):
        p = sort_with_jinn(first, h, strength=strength)
        q = sort_with_jinn(later, h, strength=strength)
        for a, b in product(sorted(q.permitted), repeat=2):
            assert isclose(p.probabilities[a] / p.probabilities[b],
                           q.probabilities[a] / q.probabilities[b],
                           rel_tol=1e-12, abs_tol=1e-12)


def test_persona_expected_score_monotonic_on_fixed_certified_support():
    s = snapshot()
    for name in ("trickster", "partisan", "reconciler", "legalist"):
        h = head(name)
        scores = {a: h.train_log_probability(ROUTES[a]) for a in s.actions}
        means = [
            sum(p * scores[a] for a, p in sort_with_jinn(s, h, strength=v).probabilities.items())
            for v in (0., .5, 1., 2., 4.)
        ]
        assert all(x <= y + 1e-12 for x, y in zip(means, means[1:]))


def test_unavailable_actions_abstain_without_renormalization():
    s = snapshot(halted=True)
    for name in ("trickster", "legalist"):
        result = sort_with_jinn(s, head(name))
        assert result.action is None
        assert not result.probabilities and result.preference_tax is None


def test_actual_head_legacy_binary_survivors_and_soft_ranking_differ():
    s = snapshot()
    assert len(s.authorized()) == 8
    for name in ("trickster", "partisan", "reconciler", "legalist"):
        h = head(name)
        legacy = frozenset(a for a in s.authorized() if h.lane(ROUTES[a]) != "reject")
        assert legacy <= s.authorized()
        out = sort_with_jinn(s, h, strength=1.)
        assert set(out.probabilities) == s.authorized()
        assert all(p > 0 for p in out.probabilities.values())
        uniform = {a: 1./len(legacy) for a in legacy}
        assert 0. <= total_variation(uniform, out.probabilities) <= 1.


def test_model_and_snapshot_fail_closed_on_invalid_inputs():
    with pytest.raises(ValueError):
        BeastSnapshot("", ("a",), BAN, {"a": evidence()})
    with pytest.raises(ValueError):
        BeastSnapshot("v", ("a", "a"), BAN, {"a": evidence()})
    with pytest.raises(ValueError):
        BeastSnapshot("v", ("a",), BAN, {})
    with pytest.raises(ValueError):
        JinnHead.from_record({"feature_names": ("a",), "classes": ("train", "hold", "reject"),
                              "weights": [[float("nan")], [0], [0]], "bias": [0, 0, 0]})
    with pytest.raises(ValueError):
        sort_with_jinn(snapshot(), head("partisan"), strength=-1.)
    with pytest.raises(ValueError):
        sort_with_jinn(snapshot(), head("partisan"), baseline={"X": 1.})
