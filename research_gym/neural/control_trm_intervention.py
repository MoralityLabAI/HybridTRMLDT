"""CPU-only action-logit intervention at ControlTRM's second GRU visit.

The frozen ControlTRM weights, encoder input, and action head are unchanged.
This helper does not provide inter-event recurrent memory or execution authority.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .control_trm import ControlTRMProposer


def _validated_features(model: ControlTRMProposer, features: Tensor) -> Tensor:
    if model.recurrence_steps != 4:
        raise ValueError("pinned ControlTRM protocol requires four recurrent visits")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("calibration intervention requires a CPU model")
    if not isinstance(features, Tensor):
        raise TypeError("features must be a tensor")
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2 or features.shape[1] != model.feature_dim:
        raise ValueError("incorrect feature shape")
    if features.device.type != "cpu" or features.dtype != next(model.parameters()).dtype:
        raise ValueError("features must match model dtype on CPU")
    if not torch.isfinite(features).all().item():
        raise ValueError("features contain non-finite values")
    return features


@torch.inference_mode()
def latent_after_two(model: ControlTRMProposer, features: Tensor) -> Tensor:
    """Read the step-two hidden state without retaining or modifying model state."""
    values = _validated_features(model, features)
    encoded = model.encoder(values)
    hidden = torch.zeros_like(encoded)
    for _ in range(2):
        hidden = model.recurrent(encoded, hidden)
    return hidden.detach().clone()


@torch.inference_mode()
def continue_from_two(model: ControlTRMProposer, features: Tensor, injected: Tensor) -> Tensor:
    """Replace h_2, then run visits three and four using recipient features."""
    values = _validated_features(model, features)
    if not isinstance(injected, Tensor):
        raise TypeError("injected state must be a tensor")
    expected = (values.shape[0], model.latent_dim)
    if tuple(injected.shape) != expected:
        raise ValueError("injected latent has incorrect shape")
    if injected.device.type != "cpu" or injected.dtype != values.dtype:
        raise ValueError("injected latent must match recipient dtype on CPU")
    if not torch.isfinite(injected).all().item():
        raise ValueError("injected latent contains non-finite values")
    encoded = model.encoder(values)
    hidden = injected.detach().clone()
    for _ in range(2, model.recurrence_steps):
        hidden = model.recurrent(encoded, hidden)
    return model.action_head(hidden)
