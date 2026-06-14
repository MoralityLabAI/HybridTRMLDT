from __future__ import annotations

from enum import Enum


class SoundnessType(str, Enum):
    """Epistemic provenance for a conflict or elimination judgment.

    LIVE: no conflict under the current analysis.
    ENV_SOUND_DEAD: unreachable under exact environment transition rules.
    MODEL_SOUND_DEAD: unreachable only after applying a model of other agents.
    EXPERIENCE_SOUND_DEAD: dead relative to replay/discovered successes only.
    UNKNOWN: insufficient evidence or unsupported judgment.
    """

    LIVE = "live"
    ENV_SOUND_DEAD = "env_sound_dead"
    MODEL_SOUND_DEAD = "model_sound_dead"
    EXPERIENCE_SOUND_DEAD = "experience_sound_dead"
    UNKNOWN = "unknown"


def is_hard(soundness: SoundnessType) -> bool:
    """Return true if the judgment is safe to use as a hard elimination."""
    return soundness == SoundnessType.ENV_SOUND_DEAD
