"""Spectral retrodiction for empirical controller-interface sheaves."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from statistics import mean
from typing import Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover - exercised without the neural extra
    raise ImportError(
        "controller_mesh_sheaf requires the optional 'neural' extra"
    ) from exc


MESH_NODES = ("proposer", "evidence", "gate", "fallback", "executor")
MESH_EDGES = (
    ("proposer", "evidence"),
    ("evidence", "gate"),
    ("gate", "executor"),
    ("proposer", "executor"),
    ("gate", "fallback"),
    ("fallback", "executor"),
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def frozen_config_sha256(config: Mapping[str, object]) -> str:
    material = {key: value for key, value in config.items() if key != "frozen_config_sha256"}
    return canonical_sha256(material)


def validate_frozen_config(config: Mapping[str, object]) -> str:
    expected = str(config.get("frozen_config_sha256") or "")
    actual = frozen_config_sha256(config)
    if not expected or expected != actual:
        raise ValueError(
            "controller-mesh sheaf config hash mismatch: "
            f"expected={expected or '<missing>'} actual={actual}"
        )
    construction = config.get("construction")
    if not isinstance(construction, Mapping):
        raise ValueError("construction is required")
    if tuple(tuple(edge) for edge in construction.get("authority_edges", ())) != MESH_EDGES:
        raise ValueError("authority topology does not match the registered mesh")
    return actual


@dataclass(frozen=True)
class Spectrum:
    eigenvalues: tuple[float, ...]
    global_section_rank: int
    low_band_rank: int
    slow_mode_rank: int
    spectral_gap: float
    algebraic_connectivity: float

    def to_jsonable(self) -> dict[str, object]:
        return {
            "eigenvalues": list(self.eigenvalues),
            "global_section_rank": self.global_section_rank,
            "low_band_rank": self.low_band_rank,
            "slow_mode_rank": self.slow_mode_rank,
            "spectral_gap": self.spectral_gap,
            "algebraic_connectivity": self.algebraic_connectivity,
        }


def _as_feature_matrix(values: Sequence[Sequence[float]] | Tensor) -> Tensor:
    matrix = torch.as_tensor(values, dtype=torch.float64)
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(1)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("module features must be a two-dimensional matrix with at least two rows")
    return matrix


def message_basis(
    values: Sequence[Sequence[float]] | Tensor,
    *,
    energy_fraction: float,
    max_rank: int,
    singular_tolerance: float,
) -> Tensor:
    """Return an orthonormal episode-function basis with an explicit constant section."""

    matrix = _as_feature_matrix(values)
    row_count = matrix.shape[0]
    constant = torch.ones((row_count, 1), dtype=torch.float64) / math.sqrt(row_count)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    centered = centered - constant @ (constant.T @ centered)
    if centered.numel() == 0 or float(centered.norm().item()) <= singular_tolerance:
        return constant
    left, singular, _ = torch.linalg.svd(centered, full_matrices=False)
    if singular.numel() == 0 or float(singular[0].item()) <= singular_tolerance:
        return constant
    keepable = int(
        (singular > float(singular[0].item()) * singular_tolerance).sum().item()
    )
    keepable = min(keepable, max_rank)
    if keepable <= 0:
        return constant
    energy = singular[:keepable].square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-18)
    retained = int((cumulative < energy_fraction).sum().item()) + 1
    retained = min(retained, keepable)
    return torch.cat((constant, left[:, :retained]), dim=1)


def _one_hot(values: Sequence[object]) -> Tensor:
    vocabulary = sorted({str(value) for value in values})
    index = {value: position for position, value in enumerate(vocabulary)}
    output = torch.zeros((len(values), max(1, len(vocabulary))), dtype=torch.float64)
    for row, value in enumerate(values):
        output[row, index[str(value)]] = 1.0
    return output


def _binary(values: Sequence[object]) -> Tensor:
    return torch.tensor([[float(bool(value))] for value in values], dtype=torch.float64)


def policy_message_matrices(
    records: Sequence[Mapping[str, object]],
    *,
    evidence_source: str,
) -> dict[str, Tensor]:
    """Build utility-label-blind module messages from final evaluation receipts."""

    if not records:
        raise ValueError("policy records are required")
    ordered = sorted(records, key=lambda row: str(row["episode_id"]))
    latent = _as_feature_matrix([row["latent"] for row in ordered])
    proposed_action = _one_hot([row["proposed_action"] for row in ordered])
    proposer = torch.cat((latent, proposed_action), dim=1)

    if evidence_source == "claim_only":
        evidence = _one_hot([row["claimed_soundness"] for row in ordered])
    elif evidence_source == "dual_channel":
        evidence = torch.cat(
            (
                _one_hot([row["claimed_soundness"] for row in ordered]),
                _one_hot([row["verified_soundness"] for row in ordered]),
                _binary([row["provenance_disagreed"] for row in ordered]),
            ),
            dim=1,
        )
    else:
        evidence = _one_hot([row["verified_soundness"] for row in ordered])

    return {
        "proposer": proposer,
        "evidence": evidence,
        "gate": _binary([row["accepted"] for row in ordered]),
        "fallback": _one_hot([row["fallback_action"] for row in ordered]),
        "executor": _one_hot([row["selected_action"] for row in ordered]),
    }


def policy_bases(
    records: Sequence[Mapping[str, object]],
    *,
    evidence_source: str,
    energy_fraction: float,
    max_rank: int,
    singular_tolerance: float,
) -> dict[str, Tensor]:
    matrices = policy_message_matrices(records, evidence_source=evidence_source)
    return {
        node: message_basis(
            matrices[node],
            energy_fraction=energy_fraction,
            max_rank=max_rank,
            singular_tolerance=singular_tolerance,
        )
        for node in MESH_NODES
    }


def sheaf_laplacian(
    bases: Mapping[str, Tensor],
    *,
    edges: Sequence[tuple[str, str]] = MESH_EDGES,
) -> Tensor:
    """Construct delta^T delta for episode-function restriction maps."""

    if set(bases) != set(MESH_NODES):
        raise ValueError("one basis is required for every registered mesh node")
    row_counts = {int(value.shape[0]) for value in bases.values()}
    if len(row_counts) != 1:
        raise ValueError("all restriction maps must share an episode response space")
    response_dim = row_counts.pop()
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for node in MESH_NODES:
        width = int(bases[node].shape[1])
        offsets[node] = (cursor, cursor + width)
        cursor += width
    delta = torch.zeros((len(edges) * response_dim, cursor), dtype=torch.float64)
    for edge_index, (left, right) in enumerate(edges):
        row_slice = slice(edge_index * response_dim, (edge_index + 1) * response_dim)
        left_start, left_stop = offsets[left]
        right_start, right_stop = offsets[right]
        delta[row_slice, left_start:left_stop] = bases[left]
        delta[row_slice, right_start:right_stop] = -bases[right]
    return delta.T @ delta


def laplacian_spectrum(
    laplacian: Tensor,
    *,
    zero_tolerance: float,
    low_band_max: float,
    slow_band_max: float,
) -> Spectrum:
    eigenvalues = torch.linalg.eigvalsh(laplacian).clamp_min(0.0)
    maximum = float(eigenvalues[-1].item()) if eigenvalues.numel() else 0.0
    if maximum > 0.0:
        eigenvalues = eigenvalues / maximum
    values = tuple(float(value) for value in eigenvalues.tolist())
    positive = [value for value in values if value > zero_tolerance]
    return Spectrum(
        eigenvalues=values,
        global_section_rank=sum(value <= zero_tolerance for value in values),
        low_band_rank=sum(value <= low_band_max for value in values),
        slow_mode_rank=sum(zero_tolerance < value <= slow_band_max for value in values),
        spectral_gap=min(positive) if positive else 0.0,
        algebraic_connectivity=(values[1] if len(values) > 1 else 0.0),
    )


def permuted_bases(bases: Mapping[str, Tensor], *, seed: int) -> dict[str, Tensor]:
    """N0: independently shuffle module transports while preserving each stalk."""

    output = {}
    for index, node in enumerate(MESH_NODES):
        generator = torch.Generator().manual_seed(seed + 104729 * index)
        order = torch.randperm(bases[node].shape[0], generator=generator)
        output[node] = bases[node][order]
    return output


def _null_summary(values: Sequence[float], observed: float) -> dict[str, float]:
    average = mean(values)
    variance = mean((value - average) ** 2 for value in values) if values else 0.0
    std = math.sqrt(variance)
    return {
        "mean": average,
        "std": std,
        "observed_minus_mean": observed - average,
        "z": (observed - average) / std if std > 1e-15 else 0.0,
        "two_sided_empirical_p": (
            1
            + sum(abs(value - average) >= abs(observed - average) for value in values)
        )
        / (len(values) + 1),
    }


def analyze_policy_sheaf(
    records: Sequence[Mapping[str, object]],
    *,
    evidence_source: str,
    energy_fraction: float,
    max_rank: int,
    singular_tolerance: float,
    zero_tolerance: float,
    low_band_max: float,
    slow_band_max: float,
    null_replicates: int,
    null_seed: int,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    bases = policy_bases(
        records,
        evidence_source=evidence_source,
        energy_fraction=energy_fraction,
        max_rank=max_rank,
        singular_tolerance=singular_tolerance,
    )
    observed = laplacian_spectrum(
        sheaf_laplacian(bases),
        zero_tolerance=zero_tolerance,
        low_band_max=low_band_max,
        slow_band_max=slow_band_max,
    )
    null_values = {
        "spectral_gap": [],
        "low_band_rank": [],
        "slow_mode_rank": [],
        "global_section_rank": [],
    }
    for replicate in range(null_replicates):
        shuffled = permuted_bases(
            bases,
            seed=null_seed + 1000003 * replicate,
        )
        null = laplacian_spectrum(
            sheaf_laplacian(shuffled),
            zero_tolerance=zero_tolerance,
            low_band_max=low_band_max,
            slow_band_max=slow_band_max,
        )
        null_values["spectral_gap"].append(null.spectral_gap)
        null_values["low_band_rank"].append(float(null.low_band_rank))
        null_values["slow_mode_rank"].append(float(null.slow_mode_rank))
        null_values["global_section_rank"].append(float(null.global_section_rank))
    observed_json = observed.to_jsonable()
    observed_json["stalk_ranks"] = {
        node: int(bases[node].shape[1]) for node in MESH_NODES
    }
    observed_json["null"] = {
        name: _null_summary(values, float(observed_json[name]))
        for name, values in null_values.items()
    }
    return observed_json, null_values


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        stop = cursor + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[cursor][1]:
            stop += 1
        rank = 0.5 * (cursor + stop - 1) + 1.0
        for index in range(cursor, stop):
            ranks[ordered[index][0]] = rank
        cursor = stop
    return ranks


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal vectors with at least two values")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_norm * right_norm
    return numerator / denominator if denominator > 1e-15 else 0.0


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def matched_null_association(
    observed_feature: Sequence[float],
    null_feature_by_policy: Sequence[Sequence[float]],
    outcome: Sequence[float],
) -> dict[str, object]:
    if not null_feature_by_policy:
        raise ValueError("matched null features are required")
    replicate_count = len(null_feature_by_policy[0])
    if any(len(values) != replicate_count for values in null_feature_by_policy):
        raise ValueError("every policy must have the same null replicate count")
    observed = spearman_correlation(observed_feature, outcome)
    null_correlations = [
        spearman_correlation(
            [values[replicate] for values in null_feature_by_policy],
            outcome,
        )
        for replicate in range(replicate_count)
    ]
    return {
        "statistic": "spearman_rho",
        "observed": observed,
        "matched_null_mean": mean(null_correlations),
        "matched_null_two_sided_p": (
            1 + sum(abs(value) >= abs(observed) for value in null_correlations)
        )
        / (replicate_count + 1),
        "null_replicates": replicate_count,
    }


def property_subspace(
    records: Sequence[Mapping[str, object]],
    *,
    rank: int,
) -> Tensor:
    """Infer a scenario-balanced sound/unsafe mean-difference subspace."""

    directions = []
    for scenario in sorted({str(row["scenario"]) for row in records}):
        rows = [row for row in records if str(row["scenario"]) == scenario]
        safe = torch.tensor(
            [row["latent"] for row in rows if bool(row["proposal_environment_sound"])],
            dtype=torch.float64,
        )
        unsafe = torch.tensor(
            [row["latent"] for row in rows if not bool(row["proposal_environment_sound"])],
            dtype=torch.float64,
        )
        if safe.numel() == 0 or unsafe.numel() == 0:
            continue
        direction = safe.mean(dim=0) - unsafe.mean(dim=0)
        norm = direction.norm()
        if float(norm.item()) > 1e-15:
            directions.append(direction / norm)
    if not directions:
        raise ValueError("property subspace needs safe and unsafe examples")
    matrix = torch.stack(directions, dim=1)
    left, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return left[:, : min(rank, left.shape[1])]


def restriction_capture(subspace: Tensor, restriction_direction: Tensor) -> float:
    direction = restriction_direction.double().flatten()
    direction = direction / direction.norm().clamp_min(1e-18)
    return float((subspace.T @ direction).square().sum().item())


def standardized_property_separation(
    records: Sequence[Mapping[str, object]],
    restriction_direction: Tensor,
) -> float:
    direction = restriction_direction.double().flatten()
    direction = direction / direction.norm().clamp_min(1e-18)
    latents = torch.tensor([row["latent"] for row in records], dtype=torch.float64)
    scores = latents @ direction
    labels = torch.tensor(
        [bool(row["proposal_environment_sound"]) for row in records],
        dtype=torch.bool,
    )
    safe = scores[labels]
    unsafe = scores[~labels]
    if len(safe) < 2 or len(unsafe) < 2:
        return 0.0
    pooled = math.sqrt(
        (
            (len(safe) - 1) * float(safe.var(unbiased=True).item())
            + (len(unsafe) - 1) * float(unsafe.var(unbiased=True).item())
        )
        / (len(safe) + len(unsafe) - 2)
    )
    if pooled <= 1e-15:
        return 0.0
    return abs(float(safe.mean().item() - unsafe.mean().item())) / pooled


def gate_margin_diagnostic(
    round0_records: Sequence[Mapping[str, object]],
    final_records: Sequence[Mapping[str, object]],
    restriction_direction: Tensor,
    *,
    bias: float,
    threshold: float,
) -> dict[str, object]:
    """Post-hoc diagnostic for translation across a frozen linear gate."""

    direction = restriction_direction.double().flatten()
    norm = direction.norm().clamp_min(1e-18)
    clipped_threshold = min(max(float(threshold), 1e-12), 1.0 - 1e-12)
    threshold_logit = math.log(clipped_threshold / (1.0 - clipped_threshold))

    def summarize(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
        latents = torch.tensor([row["latent"] for row in records], dtype=torch.float64)
        margins = (latents @ direction + float(bias) - threshold_logit) / norm
        ordered = sorted(float(value) for value in margins.tolist())
        lower_index = min(len(ordered) - 1, int(math.floor(0.05 * len(ordered))))
        return {
            "acceptance_rate": mean(value >= 0.0 for value in ordered),
            "mean_signed_margin": mean(ordered),
            "minimum_signed_margin": ordered[0],
            "p05_signed_margin": ordered[lower_index],
        }

    initial = summarize(round0_records)
    final = summarize(final_records)
    return {
        "status": "post_hoc_after_registered_kernel_test",
        "round0": initial,
        "final": final,
        "mean_margin_shift": final["mean_signed_margin"] - initial["mean_signed_margin"],
        "minimum_margin_shift": (
            final["minimum_signed_margin"] - initial["minimum_signed_margin"]
        ),
    }


def probe_sha256(weight: Tensor, *, bias: float, threshold: float) -> str:
    digest = sha256()
    digest.update(weight.detach().cpu().double().contiguous().numpy().tobytes())
    digest.update(f"bias={bias:.17g};threshold={threshold:.17g}".encode("ascii"))
    return digest.hexdigest()


def kernel_migration(
    round0_records: Sequence[Mapping[str, object]],
    final_records: Sequence[Mapping[str, object]],
    restriction_direction: Tensor,
    *,
    property_rank: int,
    null_replicates: int,
    null_seed: int,
) -> dict[str, object]:
    initial_subspace = property_subspace(round0_records, rank=property_rank)
    final_subspace = property_subspace(final_records, rank=property_rank)
    initial_capture = restriction_capture(initial_subspace, restriction_direction)
    final_capture = restriction_capture(final_subspace, restriction_direction)
    initial_separation = standardized_property_separation(
        round0_records, restriction_direction
    )
    final_separation = standardized_property_separation(final_records, restriction_direction)
    observed_capture_loss = initial_capture - final_capture
    observed_separation_loss = initial_separation - final_separation

    generator = torch.Generator().manual_seed(null_seed)
    capture_null = []
    separation_null = []
    for _ in range(null_replicates):
        direction = torch.randn(
            restriction_direction.shape,
            dtype=torch.float64,
            generator=generator,
        )
        direction = direction / direction.norm().clamp_min(1e-18)
        capture_null.append(
            restriction_capture(initial_subspace, direction)
            - restriction_capture(final_subspace, direction)
        )
        separation_null.append(
            standardized_property_separation(round0_records, direction)
            - standardized_property_separation(final_records, direction)
        )

    return {
        "round0_property_capture": initial_capture,
        "final_property_capture": final_capture,
        "kernel_occupancy_delta": observed_capture_loss,
        "round0_standardized_separation": initial_separation,
        "final_standardized_separation": final_separation,
        "standardized_separation_loss": observed_separation_loss,
        "capture_matched_random": {
            **_null_summary(capture_null, observed_capture_loss),
            "one_sided_p": (
                1 + sum(value >= observed_capture_loss for value in capture_null)
            )
            / (len(capture_null) + 1),
        },
        "separation_matched_random": {
            **_null_summary(separation_null, observed_separation_loss),
            "one_sided_p": (
                1 + sum(value >= observed_separation_loss for value in separation_null)
            )
            / (len(separation_null) + 1),
        },
    }


def summarize_policy_outcomes(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        "utility_delta_vs_proposal": mean(
            float(row["selected_utility"]) - float(row["proposal_utility"])
            for row in records
        ),
        "action_change_rate": mean(
            row["selected_action"] != row["proposed_action"] for row in records
        ),
        "acceptance_rate": mean(bool(row["accepted"]) for row in records),
    }


def effective_policy_index(
    result: Mapping[str, object],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    distinctness = result["arm_distinctness"]
    groups = distinctness["effective_policy_groups"]
    arm_to_group = {
        str(arm): str(group["effective_policy_id"])
        for group in groups
        for arm in group["arms"]
    }
    summaries = []
    for group in groups:
        topologies = {
            tuple(str(arm).split("__")[:3]) for arm in group["arms"]
        }
        summaries.append(
            {
                "effective_policy_id": group["effective_policy_id"],
                "arm_count": group["arm_count"],
                "topology_signature_count": len(topologies),
                "topology_compatible": len(topologies) == 1,
            }
        )
    return arm_to_group, summaries


def prediction_result(kernel_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_seed = {int(row["seed"]): row for row in kernel_rows}
    required = {17, 29, 43}
    if set(by_seed) != required:
        raise ValueError(f"kernel prediction requires seeds {sorted(required)}")
    values = {
        seed: float(by_seed[seed]["kernel_occupancy_delta"]) for seed in required
    }
    passed = values[29] < min(values[17], values[43])
    return {
        "registered_prediction": (
            "seed 29 has strictly less exposed-probe kernel migration than "
            "both evasion seeds 17 and 43"
        ),
        "metric": "kernel_occupancy_delta",
        "passed": passed,
        "values": {str(seed): values[seed] for seed in sorted(values)},
    }


def markdown_report(result: Mapping[str, object]) -> str:
    associations = result["associations"]
    prediction = result["kernel_migration"]["seed_29_prediction"]
    lines = [
        "# Controller-Mesh Sheaf Retrodiction",
        "",
        f"Protocol: `{result['study_id']}`.",
        "",
        "## Construction",
        "",
        (
            "The primary unit is an arm-seed policy instance. The 60 instances retain the "
            "31 behavioral equivalence classes from the source benchmark as provenance groups; "
            "topology-incompatible aliases are not averaged into one sheaf."
        ),
        "",
        (
            "Each stalk is a utility-label-blind episode-function subspace for a proposer, evidence "
            "channel, gate, fallback, or executor. Restriction maps embed those subspaces into "
            "the shared episode response space. Utility and oracle labels are revealed only "
            "after spectra are sealed."
        ),
        "",
        "## Retrodiction",
        "",
        "| Spectral feature | Outcome | Spearman rho | Matched-null p |",
        "|---|---|---:|---:|",
    ]
    for row in associations:
        lines.append(
            f"| {row['feature']} | {row['outcome']} | {row['observed']:+.3f} | "
            f"{row['matched_null_two_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Kernel Migration",
            "",
            "| Seed | Capture r0 | Capture final | Kernel migration | Random-null p |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["kernel_migration"]["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['round0_property_capture']:.4f} | "
            f"{row['final_property_capture']:.4f} | {row['kernel_occupancy_delta']:+.4f} | "
            f"{row['capture_matched_random']['one_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"The strict seed-29 prediction passed: `{prediction['passed']}`.",
            "",
            "Post-hoc frozen-gate margin diagnostic:",
            "",
            "| Seed | Pass r0 | Pass final | Mean signed-margin shift |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in result["kernel_migration"]["seeds"]:
        margin = row["posthoc_gate_margin"]
        lines.append(
            f"| {row['seed']} | {margin['round0']['acceptance_rate']:.3f} | "
            f"{margin['final']['acceptance_rate']:.3f} | "
            f"{margin['mean_margin_shift']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Matched Null",
            "",
            (
                "N0 independently permutes each module's episode transport. It preserves the "
                "authority graph, stalk rank, singular spectrum, and constant section while "
                "destroying cross-module episode compatibility."
            ),
            "",
            "## Boundary",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
