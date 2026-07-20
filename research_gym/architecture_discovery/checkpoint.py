"""Paced, atomic PyTorch checkpoints with SHA-256 sidecars."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, BinaryIO


class PacedWriter:
    def __init__(
        self,
        raw: BinaryIO,
        *,
        bytes_per_second: int = 40 * 1024 * 1024,
        chunk_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if bytes_per_second <= 0 or chunk_bytes <= 0:
            raise ValueError("pacing limits must be positive")
        self.raw = raw
        self.bytes_per_second = int(bytes_per_second)
        self.chunk_bytes = int(chunk_bytes)
        self.started = time.perf_counter()
        self.written = 0

    def write(self, value: bytes | bytearray | memoryview) -> int:
        view = memoryview(value)
        total = len(view)
        for offset in range(0, total, self.chunk_bytes):
            chunk = view[offset : offset + self.chunk_bytes]
            self.raw.write(chunk)
            self.written += len(chunk)
            expected = self.written / self.bytes_per_second
            remaining = expected - (time.perf_counter() - self.started)
            if remaining > 0:
                time.sleep(remaining)
        return total

    def flush(self) -> None:
        self.raw.flush()

    def tell(self) -> int:
        return self.raw.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.raw.seek(offset, whence)

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self.raw.fileno()


def save_paced_checkpoint(
    payload: Any,
    path: Path,
    *,
    bytes_per_second: int = 40 * 1024 * 1024,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("checkpointing requires the neural extra") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    started = time.perf_counter()
    try:
        with temporary.open("w+b") as raw:
            writer = PacedWriter(raw, bytes_per_second=bytes_per_second)
            torch.save(payload, writer)
            writer.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "bytes_per_second_cap": int(bytes_per_second),
    }
    sidecar = path.with_suffix(path.suffix + ".receipt.json")
    sidecar.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def load_verified_checkpoint(path: Path, *, map_location: str | None = None) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("checkpointing requires the neural extra") from exc
    sidecar = path.with_suffix(path.suffix + ".receipt.json")
    receipt = json.loads(sidecar.read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != receipt["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    return torch.load(path, map_location=map_location, weights_only=False)
