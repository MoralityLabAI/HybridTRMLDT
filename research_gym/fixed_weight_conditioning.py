"""Dependency-free FWC-01 synthetic state-conditioning model."""
from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    association: bool = False

def transition(state, cue, unconditioned):
    """Return (immediate response, updated recurrent state); fixed function."""
    cue = bool(cue)
    unconditioned = bool(unconditioned)
    response = unconditioned or (cue and state.association)
    updated = State(state.association or (cue and unconditioned))
    return response, updated

def authorize(typed_allowed, latent_proposal):
    """Typed gate: all actions not present in the trusted allowlist abstain."""
    return latent_proposal if latent_proposal in typed_allowed else None

def run_sequence(sequence, initial=None):
    state = initial if initial is not None else State()
    results = []
    for cue, unconditioned in sequence:
        response, state = transition(state, cue, unconditioned)
        results.append(response)
    return results, state
