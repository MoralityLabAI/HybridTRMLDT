import json
from pathlib import Path
from statistics import mean

import pytest


pytest.importorskip("torch", exc_type=ImportError)

from research_gym.analysis.measured_stalk_bridge import (
    frozen_config_sha256,
    validate_frozen_source_config,
    validate_measured_stalk,
)
from research_gym.analysis.controller_mesh_stalk_bridge import (
    external_product_eigenvalues,
    measured_graph_eigenvalues,
    shuffled_measured_edges,
    validate_bridge_config,
)
from research_gym.analysis.controller_mesh_sheaf import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_measured_stalk_source_config_is_frozen():
    path = ROOT / "configs/qwen08_l23_measured_stalk_source_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))

    assert validate_frozen_source_config(config) == frozen_config_sha256(config)
    config["selection"]["rank"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_source_config(config)


def test_compact_measured_stalk_receipt_is_self_authenticating():
    path = ROOT / "data/bridge/qwen08_l23_measured_stalk_v1.json"
    stalk = json.loads(path.read_text(encoding="utf-8"))

    assert validate_measured_stalk(stalk) == stalk["source_receipt_sha256"]
    assert stalk["activation_chunks"]["count"] == 16
    assert len(stalk["restriction_edges"]) == 10
    assert stalk["source_gate"]["checks"]["w1_trivial"] is True
    assert stalk["outcomes_consumed"] is False

    stalk["restriction_edges"][0]["restriction_weight"] = 0.0
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        validate_measured_stalk(stalk)


def test_bridge_config_and_measured_external_product_are_frozen():
    config_path = ROOT / "configs/controller_mesh_measured_stalk_bridge_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stalk = json.loads(
        (ROOT / "data/bridge/qwen08_l23_measured_stalk_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert validate_bridge_config(config) == config["frozen_config_sha256"]
    measured = measured_graph_eigenvalues(stalk)
    assert len(measured) == 8
    assert sum(value <= 1e-8 for value in measured) == 1
    product = external_product_eigenvalues((0.0, 0.25, 1.0), measured, coupling=1.0)
    assert len(product) == 24
    assert product[0] == pytest.approx(0.0)
    assert product[-1] == pytest.approx(1.0)

    config["bridge_construction"]["coupling"] = 2.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bridge_config(config)


def test_typed_stalk_null_preserves_endpoints_signs_and_weight_multisets():
    stalk = json.loads(
        (ROOT / "data/bridge/qwen08_l23_measured_stalk_v1.json").read_text(
            encoding="utf-8"
        )
    )
    shuffled = shuffled_measured_edges(stalk, seed=71)

    assert [row["edge_id"] for row in shuffled] == [
        row["edge_id"] for row in stalk["restriction_edges"]
    ]
    assert [row["restriction_sign"] for row in shuffled] == [
        row["restriction_sign"] for row in stalk["restriction_edges"]
    ]
    for edge_type in ("context", "checkpoint"):
        observed = sorted(
            row["restriction_weight"]
            for row in stalk["restriction_edges"]
            if row["edge_type"] == edge_type
        )
        null = sorted(
            row["restriction_weight"]
            for row in shuffled
            if row["edge_type"] == edge_type
        )
        assert null == observed


def test_frozen_bridge_receipts_and_qualified_negative_result():
    path = ROOT / "data/benchmarks/controller_mesh_measured_stalk_bridge_v1.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    receipts = result["stage_receipts"]

    analysis_receipt = result.pop("analysis_receipt_sha256")
    assert canonical_sha256(result) == analysis_receipt

    measured = result["measured_stalk"]
    measured_material = {
        "source_receipt_sha256": measured["source_receipt_sha256"],
        "eigenvalues": measured["graph_eigenvalues"],
        "null_eigenvalues": measured["null_graph_eigenvalues"],
    }
    assert canonical_sha256(measured_material) == receipts["measured_graph_sha256"]

    geometry_material = {
        "config_sha256": result["config_sha256"],
        "source_receipt_sha256": measured["source_receipt_sha256"],
        "measured_graph_receipt_sha256": receipts["measured_graph_sha256"],
        "task_corpus_sha256": result["task_corpus"]["sha256"],
        "features": {
            genome_id: row["combined_feature_sha256"]
            for genome_id, row in sorted(result["geometry"].items())
        },
    }
    assert canonical_sha256(geometry_material) == receipts[
        "calibration_geometry_sha256"
    ]

    prediction_material = {
        "config_sha256": result["config_sha256"],
        "source_receipt_sha256": measured["source_receipt_sha256"],
        "geometry_receipt_sha256": receipts["calibration_geometry_sha256"],
        "combined_predictor": result["combined_predictor"],
        "controller_predictor": result["controller_predictor"],
        "categorical_predictor": result["categorical_predictor"],
        "predictions": result["predictions"],
    }
    assert "outcome" not in json.dumps(prediction_material).lower()
    assert canonical_sha256(prediction_material) == receipts[
        "triple_prediction_sha256"
    ]
    assert canonical_sha256(result["discovery_outcomes"]) == receipts[
        "discovery_outcomes_sha256"
    ]
    assert canonical_sha256(result["heldout_outcomes"]) == receipts[
        "heldout_outcomes_sha256"
    ]
    assert receipts["ordering"][-2:] == [
        "combined_controller_and_categorical_predictions_sealed",
        "whole_family_heldout_outcomes_revealed",
    ]

    split = result["genome_split"]
    assert set(split["discovery_fallback_families"]).isdisjoint(
        split["heldout_fallback_families"]
    )
    bridge = result["bridge_result"]
    assert bridge["combined_rho"] == pytest.approx(0.7522522007383958)
    assert bridge["controller_rho"] == pytest.approx(0.5688022985983762)
    assert bridge["categorical_rho"] == pytest.approx(0.6401686021780727)
    assert bridge["versus_categorical"]["rho_delta"] == pytest.approx(
        0.11208359856032313
    )
    assert bridge["versus_controller"]["rho_delta"] == pytest.approx(
        0.18344990214001966
    )
    assert bridge["versus_controller"]["lower_95"] > 0.0
    assert bridge["versus_categorical"]["lower_95"] < 0.0
    assert mean(bridge["matched_null_rhos"]) == pytest.approx(
        bridge["matched_null_mean_rho"]
    )
    null_p = (
        1
        + sum(
            value >= bridge["combined_rho"]
            for value in bridge["matched_null_rhos"]
        )
    ) / (len(bridge["matched_null_rhos"]) + 1)
    assert null_p == pytest.approx(0.937984496124031)
    assert null_p == pytest.approx(bridge["matched_null_one_sided_p"])
    assert bridge["top_k_uplift_delta_vs_best_baseline"] == pytest.approx(
        -0.01124077690972225
    )
    assert bridge["gate_passed"] is False
    assert bridge["gate_checks"] == {
        "combined_rho": True,
        "rho_delta_vs_categorical": True,
        "rho_delta_vs_controller": True,
        "bootstrap_lower_vs_categorical": False,
        "bootstrap_lower_vs_controller": True,
        "matched_null": False,
        "top_k_uplift_delta": False,
    }
