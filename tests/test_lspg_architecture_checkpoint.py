from __future__ import annotations

from pathlib import Path
import time

import pytest

torch = pytest.importorskip("torch")

from research_gym.architecture_discovery.checkpoint import (  # noqa: E402
    PacedWriter,
    load_verified_checkpoint,
    save_paced_checkpoint,
)
from research_gym.architecture_discovery.training import measured_step_times  # noqa: E402


def test_paced_writer_preserves_bytes(tmp_path: Path) -> None:
    path = tmp_path / "paced.bin"
    payload = bytes(range(256)) * 32
    with path.open("w+b") as raw:
        writer = PacedWriter(raw, bytes_per_second=100_000_000, chunk_bytes=97)
        assert writer.write(payload) == len(payload)
        writer.flush()
    assert path.read_bytes() == payload


def test_checkpoint_round_trip_and_hash_guard(tmp_path: Path) -> None:
    path = tmp_path / "state.pt"
    value = {"tensor": torch.arange(16), "step": 7}
    receipt = save_paced_checkpoint(value, path, bytes_per_second=100_000_000)

    loaded = load_verified_checkpoint(path, map_location="cpu")
    assert loaded["step"] == 7
    assert torch.equal(loaded["tensor"], value["tensor"])
    assert receipt["bytes"] == path.stat().st_size

    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_checkpoint(path, map_location="cpu")


def test_resource_calibration_discards_warmup_timings() -> None:
    assert measured_step_times((9.0, 8.0, 1.0, 1.2), 2) == (1.0, 1.2)
    with pytest.raises(ValueError, match="non-negative"):
        measured_step_times((1.0,), -1)
