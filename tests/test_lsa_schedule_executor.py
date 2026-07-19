from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from research_gym.neural.schedule_executor import ScheduleExecutor  # noqa: E402


class _ResidualScalar(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, state):
        return state + self.weight * state


def test_parameter_and_state_masks_produce_different_autograd_graphs() -> None:
    block = _ResidualScalar()
    executor = ScheduleExecutor()
    inputs = torch.ones(2, requires_grad=True)

    retained = executor(
        inputs,
        modules=(block,),
        module_indices=(0, 0),
        parameter_visits=frozenset({1}),
        retained_state_edges=frozenset({0}),
    )
    retained_link = torch.autograd.grad(
        retained.output.sum(), retained.visit_outputs[0], retain_graph=True
    )[0]
    retained_parameter_grad = torch.autograd.grad(
        retained.output.sum(), block.weight
    )[0]

    truncated = executor(
        inputs,
        modules=(block,),
        module_indices=(0, 0),
        parameter_visits=frozenset({1}),
        retained_state_edges=frozenset(),
    )
    truncated_link = torch.autograd.grad(
        truncated.output.sum(), truncated.visit_outputs[0], allow_unused=True, retain_graph=True
    )[0]
    truncated_parameter_grad = torch.autograd.grad(
        truncated.output.sum(), block.weight
    )[0]

    assert retained_link is not None
    assert truncated_link is None
    assert retained_parameter_grad == pytest.approx(truncated_parameter_grad)


def test_frozen_application_still_propagates_state_gradient() -> None:
    block = _ResidualScalar()
    executor = ScheduleExecutor()
    inputs = torch.ones(2, requires_grad=True)

    execution = executor(
        inputs,
        modules=(block,),
        module_indices=(0, 0),
        parameter_visits=frozenset({1}),
        retained_state_edges=frozenset({0}),
    )

    assert torch.autograd.grad(execution.output.sum(), inputs)[0].abs().sum() > 0
