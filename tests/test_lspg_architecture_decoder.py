from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from research_gym.neural.looped_decoder import LoopedDecoderLM  # noqa: E402


def _model(word: tuple[int, ...]) -> LoopedDecoderLM:
    return LoopedDecoderLM(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        physical_modules=max(word) + 1,
        expanded_visits=len(word),
        tying="explicit",
        parameter_visits=frozenset(range(len(word))),
        retained_state_edges=frozenset(range(len(word) - 1)),
        supervision_points=(len(word) - 1,),
        alpha=1.0,
        beta=0.25,
        normalization="pre",
        carry_policy="reset",
        module_word=word,
        use_sinusoidal_positions=True,
    )


def test_explicit_schedule_word_controls_dispatch_order() -> None:
    model = _model((0, 1, 0, 2, 1, 2))

    assert model.module_indices == (0, 1, 0, 2, 1, 2)
    output = model(torch.arange(8).unsqueeze(0))
    assert len(output.visit_outputs) == 6


def test_runtime_depth_repeats_the_registered_macro_word() -> None:
    model = _model((0, 1, 0, 2, 1, 2))
    output = model(torch.arange(8).unsqueeze(0), depth_visits=9)

    assert len(output.visit_outputs) == 9


def test_matched_schedule_models_have_identical_parameters() -> None:
    torch.manual_seed(71)
    left = _model((0, 1, 0, 2, 1, 2))
    torch.manual_seed(71)
    right = _model((0, 1, 2, 0, 1, 2))

    assert left.parameter_breakdown() == right.parameter_breakdown()
    assert all(
        torch.equal(left.state_dict()[name], right.state_dict()[name])
        for name in left.state_dict()
    )
