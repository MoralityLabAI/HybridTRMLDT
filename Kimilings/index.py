"""Deterministic task inventory with public ARC-AGI-2 eval sealed by default."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLITS = ("vpd_train", "calibration", "validation")
ARC_REVISION = "f3283f727488ad98fe575ea6a5ac981e4a188e49"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(family: str, group: str, seed: int = 11) -> str:
    """Assign whole lineage groups to a split independently of row order."""
    key = f"{seed}\0{family}\0{group}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / (1 << 64)
    if value < 0.65:
        return "vpd_train"
    if value < 0.80:
        return "calibration"
    return "validation"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_num}: record must be an object")
        rows.append(row)
    return rows


def build_index(
    arc_training: Path,
    arc_evaluation: Path | None = None,
    logic_metadata: Path | None = None,
    *,
    arc_revision: str = ARC_REVISION,
    seed: int = 11,
) -> dict:
    """Index tasks without parsing or exposing final-evaluation task content.

    logic_metadata is a JSONL metadata export from the source environment with
    task_id, group_id and content_sha256; prompt/answer fields are forbidden.
    """
    if len(arc_revision) != 40 or any(c not in "0123456789abcdef" for c in arc_revision):
        raise ValueError("ARC source revision must be an exact 40-character SHA")
    if not arc_training.is_dir():
        raise FileNotFoundError(arc_training)
    rows: list[dict] = []
    for folder, is_final in ((arc_training, False), (arc_evaluation, True)):
        if folder is None:
            continue
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.glob("*.json")):
            task_id = f"arc_agi_2/{path.stem}"
            rows.append({
                "task_id": task_id,
                "family": "arc_agi_2",
                "split": "final" if is_final else stable_split("arc_agi_2", path.stem, seed),
                "group_id": path.stem,
                "content_sha256": sha256_file(path),
                "source": "arcprize/ARC-AGI-2",
            })
    if logic_metadata is not None:
        allowed = {"task_id", "group_id", "content_sha256", "task_family"}
        for item in read_jsonl(logic_metadata):
            if set(item) - allowed:
                raise ValueError("Logic metadata may contain IDs, family and hashes only")
            for key in ("task_id", "group_id", "content_sha256"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise ValueError(f"Missing logic metadata {key}")
            if len(item["content_sha256"]) != 64 or any(
                c not in "0123456789abcdef" for c in item["content_sha256"]
            ):
                raise ValueError("logic content_sha256 must be lowercase SHA-256")
            family = str(item.get("task_family", "logic"))
            if family.startswith("arc"):
                raise ValueError("Logic family cannot impersonate ARC")
            rows.append({
                "task_id": f"logic/{item['task_id']}",
                "family": family,
                "split": stable_split(family, item["group_id"], seed),
                "group_id": item["group_id"],
                "content_sha256": item["content_sha256"],
                "source": "PrimeIntellect/INTELLECT-3-RL:logic",
            })
    manifest = {
        "protocol": "KIMILINGS-01",
        "arc_revision": arc_revision,
        "logic_env_version": "0.1.2",
        "split_seed": seed,
        "records": sorted(rows, key=lambda r: r["task_id"]),
        "sealed_final_count": sum(r["split"] == "final" for r in rows),
    }
    audit_index(manifest)
    return manifest


def audit_index(manifest: dict) -> dict:
    """Reject task/group overlap and exact-content leakage across splits."""
    ids: set[str] = set()
    groups: dict[tuple[str, str], str] = {}
    hashes: dict[str, str] = {}
    by_split: dict[str, int] = {}
    for row in manifest["records"]:
        task_id = row["task_id"]
        split = row["split"]
        if split not in {*SPLITS, "final"}:
            raise ValueError(f"Unknown split: {split}")
        if task_id in ids:
            raise ValueError(f"Duplicate task ID across splits: {task_id}")
        ids.add(task_id)
        key = (row["family"], row["group_id"])
        if key in groups and groups[key] != split:
            raise ValueError(f"Lineage group crossed splits: {key}")
        groups[key] = split
        digest = row["content_sha256"]
        if digest in hashes and hashes[digest] != split:
            raise ValueError("Identical task content crossed splits")
        hashes[digest] = split
        by_split[split] = by_split.get(split, 0) + 1
        if split == "final" and row["family"] != "arc_agi_2":
            raise ValueError("Only official ARC public eval belongs in final split")
    if manifest.get("sealed_final_count") != by_split.get("final", 0):
        raise ValueError("Final count mismatch")
    return {"counts": by_split, "task_count": len(ids), "exact_hash_leakage": False,
            "group_leakage": False}


def manifest_for_task(manifest: dict) -> dict[str, dict]:
    audit_index(manifest)
    return {row["task_id"]: row for row in manifest["records"]}
