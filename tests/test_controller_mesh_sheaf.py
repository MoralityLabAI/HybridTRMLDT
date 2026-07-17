import json
from copy import deepcopy
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)

from research_gym.analysis.controller_mesh_sheaf import (
    MESH_NODES,
    analyze_policy_sheaf,
    gate_margin_diagnostic,
    kernel_migration,
    laplacian_spectrum,
    message_basis,
    permuted_bases,
    policy_bases,
    sheaf_laplacian,
    validate_frozen_config,
    canonical_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _receipt(index: int) -> dict[str, object]:
    proposed = "wait" if index % 2 == 0 else "rush"
    accepted = index in {0, 3, 4}
    fallback = "defuse" if index % 3 == 0 else "wait"
    selected = proposed if accepted else fallback
    return {
        "episode_id": f"episode-{index}",
        "latent": [float(index), float(index % 2), float((index * 3) % 5)],
        "proposed_action": proposed,
        "claimed_soundness": "env_sound_dead" if index % 2 else "unknown",
        "verified_soundness": "env_sound_dead" if accepted else "unknown",
        "provenance_disagreed": index % 3 == 0,
        "accepted": accepted,
        "fallback_action": fallback,
        "selected_action": selected,
        "proposal_utility": index / 10,
        "selected_utility": index / 9,
        "proposal_oracle_optimal": index % 2 == 0,
        "proposal_environment_sound": index in {0, 3, 4},
        "scenario": "secret" if index < 3 else "moral",
    }


def test_registered_controller_mesh_config_is_frozen():
    config = json.loads(
        (ROOT / "configs/controller_mesh_sheaf_retrodiction_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert validate_frozen_config(config) == config["frozen_config_sha256"]
    config["construction"]["low_band_max"] = 0.2
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_config(config)


def test_aligned_stalks_have_global_sections_and_normalized_spectrum():
    values = torch.tensor([[0.0], [1.0], [3.0], [2.0], [5.0], [4.0]])
    basis = message_basis(
        values,
        energy_fraction=0.95,
        max_rank=4,
        singular_tolerance=1e-9,
    )
    bases = {node: basis.clone() for node in MESH_NODES}

    spectrum = laplacian_spectrum(
        sheaf_laplacian(bases),
        zero_tolerance=1e-8,
        low_band_max=0.1,
        slow_band_max=0.25,
    )

    assert spectrum.global_section_rank == basis.shape[1]
    assert spectrum.eigenvalues[-1] == pytest.approx(1.0)
    assert min(spectrum.eigenvalues) >= 0.0


def test_n0_transport_shuffle_preserves_each_stalk_gram_matrix():
    records = [_receipt(index) for index in range(6)]
    bases = policy_bases(
        records,
        evidence_source="dual_channel",
        energy_fraction=0.95,
        max_rank=8,
        singular_tolerance=1e-9,
    )

    shuffled = permuted_bases(bases, seed=71)

    for node in MESH_NODES:
        assert torch.allclose(
            bases[node].T @ bases[node],
            shuffled[node].T @ shuffled[node],
            atol=1e-12,
            rtol=1e-12,
        )
        assert bases[node].shape == shuffled[node].shape


def test_primary_spectrum_cannot_read_utility_or_oracle_labels():
    records = [_receipt(index) for index in range(6)]
    relabeled = deepcopy(records)
    for row in relabeled:
        row["proposal_utility"] = 999.0
        row["selected_utility"] = -999.0
        row["proposal_oracle_optimal"] = not row["proposal_oracle_optimal"]
        row["proposal_environment_sound"] = not row["proposal_environment_sound"]

    kwargs = {
        "evidence_source": "exposed_frozen_probe",
        "energy_fraction": 0.95,
        "max_rank": 8,
        "singular_tolerance": 1e-9,
        "zero_tolerance": 1e-8,
        "low_band_max": 0.1,
        "slow_band_max": 0.25,
        "null_replicates": 4,
        "null_seed": 91,
    }
    original, original_null = analyze_policy_sheaf(records, **kwargs)
    changed, changed_null = analyze_policy_sheaf(relabeled, **kwargs)

    assert original == changed
    assert original_null == changed_null


def test_kernel_migration_detects_property_rotation_into_probe_kernel():
    initial = []
    final = []
    for index, safe in enumerate((False, False, True, True)):
        sign = 1.0 if safe else -1.0
        common = {
            "episode_id": f"e-{index}",
            "scenario": "one",
            "proposal_environment_sound": safe,
        }
        initial.append({**common, "latent": [sign, 0.0]})
        final.append({**common, "latent": [0.0, sign]})

    result = kernel_migration(
        initial,
        final,
        torch.tensor([1.0, 0.0]),
        property_rank=1,
        null_replicates=32,
        null_seed=101,
    )

    assert result["round0_property_capture"] == pytest.approx(1.0)
    assert result["final_property_capture"] == pytest.approx(0.0)
    assert result["kernel_occupancy_delta"] == pytest.approx(1.0)

    translated = [
        {**row, "latent": [row["latent"][0] + 3.0, row["latent"][1]]}
        for row in initial
    ]
    margin = gate_margin_diagnostic(
        initial,
        translated,
        torch.tensor([1.0, 0.0]),
        bias=0.0,
        threshold=0.5,
    )
    assert margin["mean_margin_shift"] == pytest.approx(3.0)
    assert margin["final"]["acceptance_rate"] == 1.0


def test_sealed_retrodiction_preserves_the_negative_kernel_result():
    result = json.loads(
        (ROOT / "data/benchmarks/controller_mesh_sheaf_retrodiction_v1.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = result.pop("analysis_receipt_sha256")

    assert canonical_sha256(result) == receipt
    assert result["policy_instance_count"] == 60
    assert result["behavioral_class_count"] == 31
    assert result["topology_incompatible_behavioral_class_count"] == 9
    assert result["source_receipts"]["integrity_failures"] == 0
    assert result["kernel_migration"]["seed_29_prediction"]["passed"] is False
    assert all(
        row["posthoc_gate_margin"]["final"]["acceptance_rate"] == 1.0
        for row in result["kernel_migration"]["seeds"]
    )
    association = {
        (row["feature"], row["outcome"]): row for row in result["associations"]
    }
    assert association[("spectral_gap", "utility_delta_vs_proposal")][
        "observed"
    ] == pytest.approx(-0.7225024091)
    assert association[("low_band_rank", "utility_delta_vs_proposal")][
        "observed"
    ] == pytest.approx(0.6775877102)
