"""Synthetic, CPU-only invariance checks; no training or held-out data."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from research_gym.neural.control_trm import ControlTRMProposer
from research_gym.neural.control_trm_intervention import continue_from_two, latent_after_two


def _model() -> ControlTRMProposer:
    torch.manual_seed(77)
    return ControlTRMProposer(44, 16, latent_dim=48, recurrence_steps=4).cpu().eval()


def test_second_visit_reconstructs_original_logits_and_sham() -> None:
    model = _model()
    sample = torch.linspace(-0.5, 0.5, 44).repeat(2, 1)
    baseline = model(sample)
    latent = latent_after_two(model, sample)
    assert latent.shape == (2, 48)
    assert torch.allclose(latent, baseline.latents[:, 1], atol=1e-7, rtol=1e-6)
    assert torch.allclose(continue_from_two(model, sample, latent), baseline.action_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(continue_from_two(model, sample, latent.clone()), baseline.action_logits, atol=1e-6, rtol=1e-6)


def test_donor_patch_is_local_and_rescue_restores_original_logits() -> None:
    model = _model()
    recipient = torch.linspace(-0.5, 0.5, 44).unsqueeze(0)
    donor = -recipient
    before = {k: v.clone() for k, v in model.state_dict().items()}
    baseline = model(recipient).action_logits
    own = latent_after_two(model, recipient)
    injected = latent_after_two(model, donor)
    patched = continue_from_two(model, recipient, injected)
    assert patched.shape == baseline.shape
    assert torch.isfinite(patched).all()
    rescued = continue_from_two(model, recipient, own)
    assert torch.allclose(rescued, baseline, atol=1e-6, rtol=1e-6)
    assert all(torch.equal(before[k], value) for k, value in model.state_dict().items())


@pytest.mark.parametrize("state", [torch.zeros(48), torch.zeros(1, 47), torch.zeros(2, 48), torch.full((1, 48), float("nan"))])
def test_invalid_injected_state_fails_closed(state: torch.Tensor) -> None:
    model = _model()
    with pytest.raises((ValueError, TypeError)):
        continue_from_two(model, torch.zeros(1, 44), state)


def test_incompatible_feature_and_model_recurrence_fail_closed() -> None:
    model = _model()
    with pytest.raises(ValueError):
        latent_after_two(model, torch.zeros(1, 43))
    with pytest.raises(ValueError):
        latent_after_two(model, torch.full((1, 44), float("inf")))
    short = ControlTRMProposer(44, 16, recurrence_steps=3)
    with pytest.raises(ValueError):
        latent_after_two(short, torch.zeros(1, 44))
