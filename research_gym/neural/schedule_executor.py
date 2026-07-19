"""Execute expanded visits with independent parameter and state-gradient masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
    from torch.func import functional_call
except ImportError as exc:  # pragma: no cover
    raise ImportError("schedule execution requires the optional 'neural' extra") from exc


@dataclass(frozen=True)
class ScheduleExecution:
    output: Tensor
    visit_outputs: tuple[Tensor, ...]


def _frozen_module_call(module: nn.Module, state: Tensor, **kwargs) -> Tensor:
    parameters = {name: value.detach() for name, value in module.named_parameters()}
    buffers = {name: value.detach() for name, value in module.named_buffers()}
    return functional_call(module, (parameters, buffers), (state,), kwargs)


class ScheduleExecutor(nn.Module):
    def forward(
        self,
        state: Tensor,
        *,
        modules: Sequence[nn.Module],
        module_indices: Sequence[int],
        parameter_visits: frozenset[int],
        retained_state_edges: frozenset[int],
        module_kwargs: Mapping[str, object] | None = None,
    ) -> ScheduleExecution:
        if not module_indices:
            raise ValueError("schedule requires at least one visit")
        if any(index not in range(len(modules)) for index in module_indices):
            raise ValueError("schedule references an unknown physical module")
        valid_visits = frozenset(range(len(module_indices)))
        valid_edges = frozenset(range(len(module_indices) - 1))
        if not parameter_visits.issubset(valid_visits):
            raise ValueError("parameter mask references an unknown visit")
        if not retained_state_edges.issubset(valid_edges):
            raise ValueError("state mask references an unknown edge")
        kwargs = dict(module_kwargs or {})
        outputs = []
        for visit, module_index in enumerate(module_indices):
            if visit > 0 and visit - 1 not in retained_state_edges:
                state = state.detach()
            module = modules[module_index]
            state = (
                module(state, **kwargs)
                if visit in parameter_visits
                else _frozen_module_call(module, state, **kwargs)
            )
            outputs.append(state)
        return ScheduleExecution(state, tuple(outputs))
