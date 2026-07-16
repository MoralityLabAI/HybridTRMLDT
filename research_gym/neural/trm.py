"""Small recurrent proposer with explicit latent-state access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised on installs without the extra
    raise ImportError(
        "research_gym.neural.trm requires the optional 'neural' extra"
    ) from exc

from research_gym.core.hybrid import CandidateState, HybridMode, LatticeProposal
from research_gym.core.typed_soundness import SoundnessType
from research_gym.envs.coupled_storyworld import StoryState


DEFAULT_ACTIONS = ("befriend", "investigate", "defuse", "rush", "wait")
TARGETS = ("secret_ending", "moral_optimization")
MODES = tuple(HybridMode)
SOUNDNESS_TYPES = tuple(SoundnessType)


@dataclass(frozen=True)
class TRMOutput:
    action_logits: Tensor
    mode_logits: Tensor
    conflict_logits: Tensor
    provenance_logits: Tensor
    latents: Tensor


class TRMProposer(nn.Module):
    """Dependency-isolated TRM analogue used by the gaming benchmark.

    The model encodes a target, story state, and candidate mask, then applies a
    shared GRU cell for a fixed number of recurrence steps. It exposes every
    latent and emits the repository's dependency-free LatticeProposal type.
    """

    def __init__(
        self,
        *,
        action_vocab: Sequence[str] = DEFAULT_ACTIONS,
        latent_dim: int = 256,
        recurrence_steps: int = 6,
    ) -> None:
        super().__init__()
        self.action_vocab = tuple(action_vocab)
        self.latent_dim = int(latent_dim)
        self.recurrence_steps = int(recurrence_steps)
        self.claim_threshold = 0.5
        input_dim = len(TARGETS) + 4 + len(self.action_vocab)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.Tanh(),
        )
        self.recurrent = nn.GRUCell(self.latent_dim, self.latent_dim)
        self.action_head = nn.Linear(self.latent_dim, len(self.action_vocab))
        self.mode_head = nn.Linear(self.latent_dim, len(MODES))
        self.conflict_head = nn.Linear(self.latent_dim, 2)
        self.provenance_head = nn.Linear(self.latent_dim, len(SOUNDNESS_TYPES))

    @property
    def feature_dim(self) -> int:
        return len(TARGETS) + 4 + len(self.action_vocab)

    def features_from(
        self,
        target_predicate: str,
        candidate_state: CandidateState,
        story_state: StoryState,
    ) -> Tensor:
        if target_predicate not in TARGETS:
            raise ValueError(f"unknown target predicate: {target_predicate}")
        target = [float(target_predicate == name) for name in TARGETS]
        story = [
            story_state.trust / 3.0,
            story_state.evidence / 4.0,
            story_state.heat / 4.0,
            story_state.scene / 5.0,
        ]
        domain = candidate_state.domains.get("action")
        if domain is None:
            raise KeyError("TRMProposer expects an 'action' candidate domain")
        candidates = [float(action in domain) for action in self.action_vocab]
        return torch.tensor(target + story + candidates, dtype=torch.float32)

    def forward(self, features: Tensor, *, latent_noise: Tensor | None = None) -> TRMOutput:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        encoded = self.encoder(features)
        hidden = torch.zeros_like(encoded)
        latents = []
        for _ in range(self.recurrence_steps):
            hidden = self.recurrent(encoded, hidden)
            latents.append(hidden)
        if latent_noise is not None:
            hidden = hidden + latent_noise
            latents[-1] = hidden
        stacked = torch.stack(latents, dim=1)
        return TRMOutput(
            action_logits=self.action_head(hidden),
            mode_logits=self.mode_head(hidden),
            conflict_logits=self.conflict_head(hidden),
            provenance_logits=self.provenance_head(hidden),
            latents=stacked,
        )

    def propose(
        self,
        target_predicate: str,
        candidate_state: CandidateState,
        story_state: StoryState,
        *,
        return_latents: bool = False,
        sample: bool = False,
        generator: torch.Generator | None = None,
        latent_noise_std: float = 0.0,
        force_mode: HybridMode | None = None,
    ) -> LatticeProposal | tuple[LatticeProposal, Tensor]:
        self.eval()
        features = self.features_from(target_predicate, candidate_state, story_state)
        device = next(self.parameters()).device
        features = features.to(device)
        noise = None
        if latent_noise_std > 0:
            noise = torch.randn(
                (1, self.latent_dim),
                generator=generator,
                device=device,
            ) * float(latent_noise_std)
        with torch.no_grad():
            output = self.forward(features, latent_noise=noise)
            action_logits = output.action_logits[0].clone()
            domain = candidate_state.domains["action"]
            for index, action in enumerate(self.action_vocab):
                if action not in domain:
                    action_logits[index] = -torch.inf
            if sample:
                action_index = int(
                    torch.multinomial(
                        torch.softmax(action_logits, dim=0),
                        1,
                        generator=generator,
                    ).item()
                )
            else:
                action_index = int(action_logits.argmax().item())
            mode = force_mode or MODES[int(output.mode_logits[0].argmax().item())]
            env_index = SOUNDNESS_TYPES.index(SoundnessType.ENV_SOUND_DEAD)
            unknown_index = SOUNDNESS_TYPES.index(SoundnessType.UNKNOWN)
            env_unknown = torch.softmax(
                output.provenance_logits[0, [env_index, unknown_index]],
                dim=0,
            )
            env_claim_score = float(env_unknown[0].item())
            soundness = (
                SoundnessType.ENV_SOUND_DEAD
                if env_claim_score >= self.claim_threshold
                else SoundnessType.UNKNOWN
            )
            selected_action = self.action_vocab[action_index]
            final_latent = output.latents[0, -1].detach().cpu()
            proposal = LatticeProposal(
                proposed_state=candidate_state,
                soundness=soundness,
                mode=mode,
                conflict_score=float(
                    torch.softmax(output.conflict_logits[0], dim=0)[1].item()
                ),
                metadata={
                    "selected_action": selected_action,
                    "target_predicate": target_predicate,
                    "story_state": story_state.to_dict(),
                    "latent": final_latent.tolist(),
                    "env_claim_score": env_claim_score,
                    "claim_threshold": self.claim_threshold,
                },
            )
        if return_latents:
            return proposal, output.latents[0].detach().cpu()
        return proposal
