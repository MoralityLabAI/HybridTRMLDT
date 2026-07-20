from __future__ import annotations

from hashlib import sha256

from research_gym.integrity import (
    accepted_file_sha256s,
    canonical_file_sha256,
    verify_file_sha256,
)


def test_text_hash_is_invariant_to_checkout_line_endings(tmp_path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"value":1}\n{"value":2}\n')
    crlf.write_bytes(b'{"value":1}\r\n{"value":2}\r\n')

    assert canonical_file_sha256(lf) == canonical_file_sha256(crlf)
    assert verify_file_sha256(crlf, canonical_file_sha256(lf))


def test_text_verifier_accepts_already_sealed_raw_digest(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    payload = b'{"value":1}\r\n'
    path.write_bytes(payload)
    legacy_digest = sha256(payload).hexdigest()

    assert legacy_digest in accepted_file_sha256s(path)
    assert verify_file_sha256(path, legacy_digest)


def test_binary_hash_remains_byte_exact(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    payload = b"header\r\npayload\r"
    path.write_bytes(payload)

    expected = sha256(payload).hexdigest()
    normalized = sha256(payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
    assert canonical_file_sha256(path) == expected
    assert accepted_file_sha256s(path) == frozenset({expected})
    assert not verify_file_sha256(path, normalized)
