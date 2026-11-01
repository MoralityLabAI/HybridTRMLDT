"""Leak-controlled, reproducible supervised frame export for RIVET stage 1.

Only public pre-decision fields enter model inputs. Targets are hypotheses for
supervised imitation; independent mechanics verification must still authorize
any durable lattice update. Correction-stage traces require separate genuine
verifier rollouts and are intentionally absent here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from research_gym.core.frames import Frame, read_frame_jsonl

SCHEMA_VERSION = "rivet_stage1_sft_v1"
INSTRUCTION = (
    "Given the public state and candidate operation, propose the next typed "
    "decision as JSON. Your answer is an unverified hypothesis; an independent "
    "mechanics checker has sole authority to certify a lattice update."
)
SPLITS = ("train", "calibration", "eval")
INPUT_FIELDS = ("family", "input_state", "operation")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def public_input(frame: Frame) -> dict[str, Any]:
    """Explicit input-field allowlist. Never serialize Frame.to_jsonable()."""
    return {
        "family": frame.family,
        "input_state": dict(frame.input_state),
        "operation": frame.operation,
    }


def proposal_target(frame: Frame) -> dict[str, Any]:
    """Gold supervision is not an environment-issued certificate."""
    return {
        "label": frame.label,
        "predicted_soundness_type": frame.soundness_type.value,
        "proposed_output_state": dict(frame.output_state),
    }


def group_digest(frame: Frame) -> str:
    """Equivalent public states/operations always remain in one split."""
    return hashlib.sha256(_canonical(public_input(frame)).encode("utf-8")).hexdigest()


def split_for(frame: Frame, *, holdout_families: Sequence[str] = ()) -> str:
    if frame.family in holdout_families:
        return "eval"
    bucket = int(group_digest(frame)[:12], 16) % 10_000
    if bucket < 8000:
        return "train"
    if bucket < 9000:
        return "calibration"
    return "eval"


def sft_record(frame: Frame, *, holdout_families: Sequence[str] = ()) -> dict[str, Any]:
    split = split_for(frame, holdout_families=holdout_families)
    return {
        "instruction": INSTRUCTION,
        "input": _canonical(public_input(frame)),
        "output": _canonical(proposal_target(frame)),
        "metadata": {
            "id": frame.id,
            "family": frame.family,
            "source": frame.source,
            "group_sha256": group_digest(frame),
            "split": split,
            "authority": "supervised_prediction_only",
            "schema_version": SCHEMA_VERSION,
        },
    }


def read_frames(source: Path) -> list[Frame]:
    paths = [source] if source.is_file() else sorted(source.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no common Frame JSONL found under {source}")
    frames: list[Frame] = []
    for path in paths:
        frames.extend(read_frame_jsonl(path))
    return frames


def export(
    frames: Sequence[Frame],
    out_dir: Path,
    *,
    holdout_families: Sequence[str] = (),
) -> dict[str, Any]:
    if not frames:
        raise ValueError("cannot export an empty frame collection")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [sft_record(frame, holdout_families=holdout_families) for frame in frames]
    group_splits: dict[str, str] = {}
    counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    for record in records:
        meta = record["metadata"]
        group, split = meta["group_sha256"], meta["split"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"split leakage: {group}")
        group_splits[group] = split
        counts[split] += 1
        family_counts[f"{split}:{meta['family']}"] += 1

    for split in SPLITS:
        dest = out_dir / f"{split}.jsonl"
        with dest.open("w", encoding="utf-8") as handle:
            for record in records:
                if record["metadata"]["split"] == split:
                    handle.write(_canonical(record) + "\n")
        hashes[dest.name] = hashlib.sha256(dest.read_bytes()).hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_allowlist": list(INPUT_FIELDS),
        "output_role": "proposal_supervision_unverified",
        "holdout_families": sorted(set(holdout_families)),
        "split_rule": "sha256(canonical(public_input))[:12] mod 10000; 0:7999 train; 8000:8999 calibration; 9000:9999 eval; holdout families forced eval",
        "counts": {name: counts[name] for name in SPLITS},
        "family_counts": dict(sorted(family_counts.items())),
        "unique_groups": len(group_splits),
        "file_sha256": hashes,
        "notes": [
            "Frame.metadata is excluded from model inputs because it can contain counterfactual witnesses.",
            "An SFT label is not an independently verified environment-sound certificate.",
            "Real REPAIR trajectories must be collected from a separate mechanics/verifier runner.",
            "A family holdout in eval is also excluded from calibration.",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export leakage-controlled RIVET stage-1 SFT.")
    parser.add_argument("--source", type=Path, default=Path("data/generated"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft/rivet_stage1_v1"))
    parser.add_argument("--holdout-family", action="append", default=[])
    args = parser.parse_args()
    manifest = export(
        read_frames(args.source),
        args.out_dir,
        holdout_families=args.holdout_family,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
