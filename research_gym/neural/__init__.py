"""Optional neural components for the hybrid research gym.

Importing this package does not import torch. Consumers that need the neural
extra should import from ``research_gym.neural.trm`` or related modules.
"""

from __future__ import annotations

from importlib.util import find_spec


def torch_available() -> bool:
    return find_spec("torch") is not None


__all__ = ["torch_available"]

