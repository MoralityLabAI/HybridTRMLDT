from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from research_gym.neural.looped_decoder import LoopedDecoderLM  # noqa: E402
from research_gym.prognostics.planner import LoopSchedulePlanner  # noqa: E402
from research_gym.prognostics.schemas import ResourceProfile  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _proposal():
    planner = LoopSchedulePlanner(
        base_receipt=ROOT / "data/benchmarks/lsa_v0_receipt.json",
        scale_ladder=_load("configs/lsa/scale_ladder_v0.json"),
        mutation_space=_load("configs/lsa/mutation_space_v0.json"),
        resource_profile=ResourceProfile(2_147_483_648, 1_572_864_000, 50, 52_428_800, 1800, 12),
        promotion_policy=_load("configs/lsa/promotion_policy_v0.json"),
        code_commit="test",
    )
    return planner.propose(12)[0].to_dict()


def test_looped_decoder_matches_solved_unique_parameter_count() -> None:
    proposal = _proposal()
    model = LoopedDecoderLM.from_proposal(proposal)

    assert model.parameter_breakdown()["unique_parameters"] == proposal["model"]["unique_parameters"]
    assert model.parameter_breakdown()["embedding_parameters"] == proposal["model"]["embedding_parameters"]


def test_looped_decoder_produces_causal_logits_and_visit_states() -> None:
    proposal = _proposal()
    model = LoopedDecoderLM.from_proposal(proposal)
    tokens = torch.arange(16).remainder(model.vocab_size).unsqueeze(0)

    output = model(tokens)

    assert output.logits.shape == (1, 16, model.vocab_size)
    assert len(output.visit_outputs) == proposal["mutation"]["expanded_visits"]
    assert not hasattr(model, "lm_head")
