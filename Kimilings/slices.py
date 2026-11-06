"""Offline byte-level preflight for a normalized 16-index Kimi slice receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Kimilings.index import sha256_file

SLICE_REPO = "AlephFunk/Kimi-K3.1-Pro-expert-slices-0-15"
SLICE_COMMIT = "387322684297ff5f014d772f4ac6549c6d94a77d"


def verify_slice_receipt(receipt: dict, root: Path | None = None) -> dict:
    """Verify 16 bank bundles and optional exact local bytes.

    An independent normalized receipt is required. Metadata inspection alone
    is reported as UNVERIFIED, even if it contains hashes.
    """
    if receipt.get("hf_repo") != SLICE_REPO or receipt.get("hf_commit") != SLICE_COMMIT:
        raise ValueError("Wrong Kimi slice repository revision")
    if receipt.get("tensor_count") != 8832:
        raise ValueError("Expected 8832 independently extracted tensors")
    bundles = receipt.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 16:
        raise ValueError("Exactly 16 bundle entries are required")
    indices = set()
    paths = set()
    total_bytes = 0
    for bundle in bundles:
        index = bundle.get("expert_index")
        path = bundle.get("path")
        digest = bundle.get("sha256")
        size = bundle.get("size_bytes")
        if type(index) is not int or index not in range(16) or index in indices:
            raise ValueError("Bundle indices must cover 0..15 exactly once")
        indices.add(index)
        if not isinstance(path, str) or not path or path in paths:
            raise ValueError("Duplicate or empty bundle path")
        paths.add(path)
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise ValueError("Unsafe bundle path")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError("Invalid bundle SHA-256")
        if type(size) is not int or size <= 0:
            raise ValueError("Invalid bundle size")
        total_bytes += size
        if root is not None:
            absolute = (root / relative).resolve()
            if not absolute.is_relative_to(root.resolve()):
                raise ValueError("Bundle symlink escapes source root")
            if absolute.stat().st_size != size or sha256_file(absolute) != digest:
                raise ValueError(f"Bundle bytes mismatch: {index}")
    return {
        "source_repo": SLICE_REPO, "source_commit": SLICE_COMMIT,
        "bundle_count": 16, "tensor_count": 8832,
        "total_expected_bytes": total_bytes,
        "status": "local_sha256_and_size_verified" if root is not None
                  else "metadata_inspected_bytes_unverified",
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="Local path containing the bundles")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    report = verify_slice_receipt(receipt, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])


if __name__ == "__main__":
    main()
