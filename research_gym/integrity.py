"""Cross-platform hashing for receipt-bearing files."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


TEXT_SUFFIXES = frozenset(
    {
        ".csv",
        ".json",
        ".jsonl",
        ".md",
        ".py",
        ".sha256",
        ".svg",
        ".tex",
        ".toml",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def is_text_artifact(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TEXT_SUFFIXES


def normalize_text_newlines(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_file_bytes(path: str | Path) -> bytes:
    target = Path(path)
    payload = target.read_bytes()
    return normalize_text_newlines(payload) if is_text_artifact(target) else payload


def raw_file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical_file_sha256(path: str | Path) -> str:
    return sha256(canonical_file_bytes(path)).hexdigest()


def accepted_file_sha256s(path: str | Path) -> frozenset[str]:
    """Return canonical plus legacy raw digests for text, raw only for binary."""

    raw = raw_file_sha256(path)
    if not is_text_artifact(path):
        return frozenset({raw})
    return frozenset({raw, canonical_file_sha256(path)})


def verify_file_sha256(path: str | Path, expected: str) -> bool:
    """Verify canonical text hashes while preserving already-sealed raw receipts."""

    return expected.lower() in accepted_file_sha256s(path)
