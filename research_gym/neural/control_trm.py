"""Small recurrent proposer for long-context control-task public features."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("research_gym.neural.control_trm requires the optional 'neural' extra") from exc


@dataclass(frozen=True)
class ControlTRMOutput:
    action_logits: Tensor
    confidence: Tensor
    provenance_logits: Tensor
    latents: Tensor


class ControlTRMProposer(nn.Module):
    """Gym TRM analogue with tied recurrent visits and explicit latent access."""

    def __init__(self, feature_dim: int, action_count: int, *, latent_dim: int = 48, recurrence_steps: int = 4):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_count = int(action_count)
        self.latent_dim = int(latent_dim)
        self.recurrence_steps = int(recurrence_steps)
        self.encoder = nn.Sequential(nn.Linear(feature_dim, latent_dim), nn.Tanh())
        self.recurrent = nn.GRUCell(latent_dim, latent_dim)
        self.action_head = nn.Linear(latent_dim, action_count)
        self.confidence_head = nn.Linear(latent_dim, 1)
        self.provenance_head = nn.Linear(latent_dim, 2)

    def forward(self, features: Tensor) -> ControlTRMOutput:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        encoded = self.encoder(features)
        hidden = torch.zeros_like(encoded)
        latents = []
        for _ in range(self.recurrence_steps):
            hidden = self.recurrent(encoded, hidden)
            latents.append(hidden)
        return ControlTRMOutput(
            action_logits=self.action_head(hidden),
            confidence=torch.sigmoid(self.confidence_head(hidden)).squeeze(-1),
            provenance_logits=self.provenance_head(hidden),
            latents=torch.stack(latents, dim=1),
        )
