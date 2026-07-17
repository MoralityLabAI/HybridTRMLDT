"""Measured Qwen stalk extraction and controller-mesh bridge analysis."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from research_gym.analysis.controller_mesh_sheaf import canonical_sha256


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_config_sha256(config: Mapping[str, object]) -> str:
    material = {
        key: value for key, value in config.items() if key != "frozen_config_sha256"
    }
    return canonical_sha256(material)


def validate_frozen_source_config(config: Mapping[str, object]) -> str:
    expected = str(config.get("frozen_config_sha256") or "")
    actual = frozen_config_sha256(config)
    if not expected or expected != actual:
        raise ValueError(
            "measured-stalk source config hash mismatch: "
            f"expected={expected or '<missing>'} actual={actual}"
        )
    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("measured-stalk selection is required")
    if selection.get("site") != "model.layers.23" or int(selection.get("rank", 0)) != 1:
        raise ValueError("only the frozen layer-23 rank-one source is admissible")
    return actual


def _resolve_source_files(
    config: Mapping[str, object],
    *,
    root_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Path]:
    roots = {
        str(name): Path(path)
        for name, path in dict(config["source_roots"]).items()
    }
    if root_overrides:
        roots.update({str(name): Path(path) for name, path in root_overrides.items()})
    resolved = {}
    for name, entry in dict(config["files"]).items():
        root_name = str(entry["root"])
        if root_name not in roots:
            raise ValueError(f"unknown measured-stalk source root: {root_name}")
        path = (roots[root_name] / str(entry["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"measured-stalk source is missing: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"measured-stalk source hash mismatch for {name}: "
                f"expected={entry['sha256']} actual={actual}"
            )
        resolved[str(name)] = path
    return resolved


def _selected_chunk_rows(
    index_paths: Mapping[str, Path],
    selection: Mapping[str, object],
) -> tuple[list[dict[str, object]], int]:
    selected = []
    total_bytes = 0
    expected_states = set(map(str, selection["states"]))
    expected_halves = set(map(str, selection["halves"]))
    expected_shards = set(map(str, selection["context_shards"]))
    for state, index_path in index_paths.items():
        value = json.loads(index_path.read_text(encoding="utf-8"))
        if value.get("state_id") != state or state not in expected_states:
            raise ValueError(f"capture index has an unexpected state: {state}")
        for row in value.get("chunks", ()):
            if row.get("site_id") != selection["site"]:
                continue
            if row.get("state_id") != state:
                raise ValueError("capture chunk state differs from its index")
            if row.get("half") not in expected_halves:
                raise ValueError("capture chunk half differs from the frozen universe")
            if row.get("context_shard") not in expected_shards:
                raise ValueError("capture context shard differs from the frozen universe")
            path = (index_path.parent / str(row["path"])).resolve()
            if not path.is_file() or sha256_file(path) != row.get("sha256"):
                raise ValueError(f"measured activation chunk changed: {row.get('chunk_id')}")
            total_bytes += path.stat().st_size
            selected.append(
                {
                    key: row[key]
                    for key in (
                        "chunk_id",
                        "state_id",
                        "site_id",
                        "context_shard",
                        "half",
                        "path",
                        "sha256",
                        "row_count",
                        "ambient_dimension",
                        "dtype",
                    )
                }
            )
    selected.sort(key=lambda row: str(row["chunk_id"]))
    if len(selected) != int(selection["expected_chunk_count"]):
        raise ValueError("measured activation chunk count differs from registration")
    if canonical_sha256(selected) != selection["expected_chunk_manifest_sha256"]:
        raise ValueError("measured activation chunk manifest differs from registration")
    return selected, total_bytes


def _selected_edge_rows(
    edge_path: Path, selection: Mapping[str, object]
) -> list[dict[str, object]]:
    raw = [
        json.loads(line)
        for line in edge_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prefix = f"{selection['site']}:"
    selected = []
    for row in raw:
        if not str(row.get("edge_id", "")).startswith(prefix):
            continue
        construction = float(
            row["construction_lineage"]["worst_direction_retention"]
        )
        validation = float(
            row["geometry_validation_lineage"]["worst_direction_retention"]
        )
        construction_det = float(row["construction_transport_det"])
        validation_det = float(row["geometry_validation_transport_det"])
        selected.append(
            {
                "edge_id": row["edge_id"],
                "source_node": row["source_node"],
                "target_node": row["target_node"],
                "rank": int(row["rank"]),
                "construction_lineage": construction,
                "validation_lineage": validation,
                "construction_transport_det": construction_det,
                "validation_transport_det": validation_det,
                "transport_hash": row["transport_hash"],
            }
        )
    selected.sort(key=lambda row: str(row["edge_id"]))
    if len(selected) != int(selection["expected_edge_count"]):
        raise ValueError("measured restriction edge count differs from registration")
    if canonical_sha256(selected) != selection["expected_edge_payload_sha256"]:
        raise ValueError("measured restriction edge payload differs from registration")
    return selected


def _validate_source_gates(
    repository_receipt: Mapping[str, object],
    stage_result: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    if repository_receipt.get("outcomes_consumed") is not False:
        raise ValueError("repository source consumed outcomes")
    if stage_result.get("outcomes_consumed") is not False:
        raise ValueError("stage source consumed outcomes")
    if repository_receipt.get("weight_mutation") is not False:
        raise ValueError("repository source performed weight mutation")
    if stage_result.get("weight_mutation") is not False:
        raise ValueError("stage source performed weight mutation")
    site_rows = [
        row
        for row in stage_result["main"]["site_results"]
        if row.get("site") == selection["site"]
    ]
    if len(site_rows) != 1:
        raise ValueError("frozen measured site is absent or duplicated")
    row = site_rows[0]
    phase = row["phase_at_registered_floor"]
    checks = {
        "repository_main_gate_passed": repository_receipt["main_gate"]["passed"]
        is True,
        "stage_main_gate_passed": stage_result["main"]["main_gate_passed"] is True,
        "rank": int(row["rank"]) == int(selection["rank"]),
        "phase": phase["phase"] == "coherent",
        "w1_trivial": phase["syndrome"]["w1_trivial"] is True,
        "outcomes_not_consumed": stage_result.get("outcomes_consumed") is False,
        "weight_mutation_absent": stage_result.get("weight_mutation") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"measured-stalk source gate failed: {checks}")
    return {
        "checks": checks,
        "beta_1": int(phase["beta_1"]),
        "syndrome": list(phase["syndrome"]["syndrome"]),
        "w1_gate_passed": row["w1_gate_passed"] is True,
        "admitted_generator_count": int(row["admitted_generator_count"]),
    }


def extract_measured_stalk(
    config: Mapping[str, object],
    *,
    root_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    config_sha256 = validate_frozen_source_config(config)
    files = _resolve_source_files(config, root_overrides=root_overrides)
    repository_receipt = json.loads(
        files["repository_receipt"].read_text(encoding="utf-8")
    )
    stage_result = json.loads(files["stage_result"].read_text(encoding="utf-8"))
    selection = config["selection"]
    gate = _validate_source_gates(repository_receipt, stage_result, selection)
    chunks, chunk_bytes = _selected_chunk_rows(
        {
            "base": files["base_index"],
            "naive_qlora": files["naive_index"],
        },
        selection,
    )
    edges = _selected_edge_rows(files["edge_receipts"], selection)
    restrictions = []
    for row in edges:
        construction_det = float(row["construction_transport_det"])
        validation_det = float(row["validation_transport_det"])
        if construction_det * validation_det <= 0.0:
            raise ValueError("construction and validation transport signs disagree")
        restrictions.append(
            {
                **row,
                "edge_type": (
                    "checkpoint" if ":precision:" in str(row["edge_id"]) else "context"
                ),
                "restriction_weight": min(
                    float(row["construction_lineage"]),
                    float(row["validation_lineage"]),
                ),
                "restriction_sign": 1 if validation_det > 0.0 else -1,
            }
        )
    result: dict[str, object] = {
        "schema": "qwen08_l23_measured_stalk_v1",
        "source_id": config["source_id"],
        "source_config_sha256": config_sha256,
        "model_id": selection["model_id"],
        "site": selection["site"],
        "rank": int(selection["rank"]),
        "source_files": {
            name: {
                "root": config["files"][name]["root"],
                "path": config["files"][name]["path"],
                "sha256": config["files"][name]["sha256"],
            }
            for name in sorted(files)
        },
        "activation_chunks": {
            "count": len(chunks),
            "bytes": chunk_bytes,
            "manifest_sha256": canonical_sha256(chunks),
            "chunks": chunks,
        },
        "restriction_edges": restrictions,
        "source_gate": gate,
        "outcomes_consumed": False,
        "generation": False,
        "gradients": False,
        "weight_mutation": False,
        "claim_boundary": config["claim_boundary"],
    }
    result["source_receipt_sha256"] = canonical_sha256(result)
    return result


def validate_measured_stalk(stalk: Mapping[str, object]) -> str:
    if stalk.get("schema") != "qwen08_l23_measured_stalk_v1":
        raise ValueError("unexpected measured-stalk schema")
    expected = str(stalk.get("source_receipt_sha256") or "")
    material = {
        key: value for key, value in stalk.items() if key != "source_receipt_sha256"
    }
    actual = canonical_sha256(material)
    if not expected or expected != actual:
        raise ValueError("measured-stalk receipt hash mismatch")
    if stalk.get("outcomes_consumed") is not False:
        raise ValueError("measured stalk is outcome-bearing")
    return actual
