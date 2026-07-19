from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from research_gym.attestation.provenance_gate import (
    CLAIM_BOUNDARY,
    RSIAttestationBackend,
    canonical_sha256,
    run_rsi_conformance,
    validate_registration,
)
from research_gym.benchmarks.attested_provenance_gate_bench import (
    run_attested_provenance_benchmark,
    verify_local_sources,
)


ROOT = Path(__file__).resolve().parents[2]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _rate(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _range(values: Sequence[float]) -> str:
    return f"{100.0 * min(values):.1f}% to {100.0 * max(values):.1f}%"


def month1_summary(result: Mapping[str, object]) -> str:
    endpoints = result["primary_endpoints_per_seed"]
    tampers = result["negative_controls"]["tamper_controls"]
    blind = [row["tamper_class"] for row in tampers if row["status"] != "DETECTED"]
    suppressions = [
        float(row["false_claim_acceptance"]["absolute_suppression"])
        for row in endpoints
    ]
    acceptance_costs = [
        float(row["aligned_usefulness"]["acceptance_rate_cost"])
        for row in endpoints
    ]
    utility_costs = [
        float(row["aligned_usefulness"]["executed_utility_cost"])
        for row in endpoints
    ]
    if blind:
        first = f"The attested gate had blind spots: {', '.join(blind)}."
    else:
        first = (
            "The attested gate suppressed emergent false-provenance acceptance by "
            f"{_range(suppressions)} across the three separately reported seeds."
        )
    second = (
        "The aligned-usefulness acceptance-rate cost ranged from "
        f"{_range(acceptance_costs)}, while the signed executed-utility cost ranged from "
        f"{min(utility_costs):+.4f} to {max(utility_costs):+.4f}."
    )
    third = (
        "This result is limited to anchor-bound claim/evidence channels in the frozen toy "
        "storyworld and does not certify neural feature identity or oversight sufficiency."
    )
    return " ".join((first, second, third))


def report_markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Attested Provenance Gate x Gaming Benchmark",
        "",
        "## Result",
        "",
        month1_summary(result),
        "",
        "The co-primary endpoints remain per-seed. No endpoint is pooled across seeds.",
        "",
        "| Seed | False acceptance, claim-only | False acceptance, attested | Suppression | True-proposal acceptance cost | Executed-utility cost |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["primary_endpoints_per_seed"]:
        false = row["false_claim_acceptance"]
        useful = row["aligned_usefulness"]
        lines.append(
            "| {seed} | {base} ({base_n}/{base_d}) | {att} ({att_n}/{att_d}) | {supp} | {accept_cost} | {utility_cost:+.4f} |".format(
                seed=row["seed"],
                base=_rate(float(false["claim_only"]["acceptance_rate"])),
                base_n=false["claim_only"]["accepted_count"],
                base_d=false["claim_only"]["attempt_count"],
                att=_rate(float(false["attested_claim"]["acceptance_rate"])),
                att_n=false["attested_claim"]["accepted_count"],
                att_d=false["attested_claim"]["attempt_count"],
                supp=_rate(float(false["absolute_suppression"])),
                accept_cost=_rate(float(useful["acceptance_rate_cost"])),
                utility_cost=float(useful["executed_utility_cost"]),
            )
        )
    lines.extend(
        [
            "",
            "Costs are signed `claim_only - attested_claim`; negative values favor the attested arm.",
            "The executed-utility endpoint is conditioned on each arm's truly environment-sound proposals, so adaptation-induced proposal-distribution changes remain part of the whole-system comparison.",
            "",
            "## Construction",
            "",
            "Each `ENV_SOUND_DEAD` claim requests a receipt binding the exact storyworld state hash, mechanics-source identity, horizon, action, claim type, anchor-record hash, and sealed registry root. The gate delegates lineage and holonomy certification to the hash-pinned RSITopology `AnchorRegistry` and requires authorization for `signed_intervention`. Unsafe claims can still emerge in the proposer report channel, but no trusted registry entry exists for their state/action payload, so their envelopes are marked unissued and denied.",
            "",
            "This construction assumes the exact-mechanics issuer and registry root are trusted. It tests whether provenance binding blocks report-channel gaming; it does not show that identity attestation independently discovers semantic truth.",
            "",
            "## Controls",
            "",
            f"- RSITopology golden-vector conformance: `{result['conformance']['golden_vector_count']}` replayed, `{result['conformance']['golden_vector_failure_count']}` failures.",
            f"- No-attestation v1 cell replay: `{result['negative_controls']['no_attestation_v1_replay']['byte_semantics_reproduced']}`; final summary reproduced: `{result['negative_controls']['no_attestation_v1_replay']['final_summary_reproduced']}`.",
            f"- Identical fallback zero proposal/utility delta: `{result['negative_controls']['identical_fallback_zero_delta']}` over `{result['negative_controls']['identical_fallback_row_count']}` decisions.",
            f"- Split group overlap: `{sum(row['group_overlap'] for row in result['split_receipts'])}`.",
            "",
            "| Registered tamper | Outcome | Detector evidence |",
            "|---|---|---|",
        ]
    )
    for row in result["negative_controls"]["tamper_controls"]:
        lines.append(
            f"| `{row['tamper_class']}` | **{row['status']}** | `{', '.join(row['failures'])}` |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- frozen config SHA-256: `{result['config_sha256']}`",
            f"- canonical decision JSONL SHA-256: `{result['artifacts']['records_sha256']}`",
            f"- decision rows: `{result['record_count']}`",
            f"- decision receipt failures: `{result['artifacts']['decision_receipt_failure_count']}`",
            f"- attestation decision receipt failures: `{result['artifacts']['attestation_decision_receipt_failure_count']}`",
            f"- registry count: `{len(result['registries'])}`",
            "",
            "## Claim Boundary",
            "",
            CLAIM_BOUNDARY,
            "",
            "Per-seed variance is a finding, not noise; single-seed conclusions are inadmissible.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "attested_provenance_gate_v1.json")
    parser.add_argument("--rsi-root", type=Path, default=Path(r"C:\projects\RSITopology"))
    parser.add_argument("--conformance-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    registration = json.loads(args.config.read_text(encoding="utf-8"))
    config_sha256 = validate_registration(registration)
    verify_local_sources(ROOT, registration)
    conformance_path = ROOT / "data" / "benchmarks" / "attested_provenance_gate_v1_conformance.json"
    if args.conformance_only:
        backend = RSIAttestationBackend(args.rsi_root, registration["source_integrity"])
        receipt = run_rsi_conformance(backend, registration["source_integrity"])
        receipt["config_sha256"] = config_sha256
        _write(conformance_path, _json_bytes(receipt))
        if not receipt["passed"]:
            raise SystemExit("RSITopology conformance failed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    suffix = "_smoke" if args.smoke else ""
    stem = f"attested_provenance_gate_v1{suffix}"
    registry_dir = ROOT / "data" / "benchmarks" / f"{stem}_registries"
    backend = RSIAttestationBackend(args.rsi_root, registration["source_integrity"])
    preflight_conformance = run_rsi_conformance(backend, registration["source_integrity"])
    preflight_conformance["config_sha256"] = config_sha256
    _write(conformance_path, _json_bytes(preflight_conformance))
    if not preflight_conformance["passed"]:
        raise SystemExit("RSITopology conformance failed")
    result, records, conformance = run_attested_provenance_benchmark(
        registration,
        root=ROOT,
        rsi_root=args.rsi_root,
        registry_dir=registry_dir,
        smoke=args.smoke,
    )
    records_payload = _jsonl_bytes(records)
    records_sha256 = sha256(records_payload).hexdigest()
    decision_failures = sum(not bool(row["integrity_ok"]) for row in records)
    attestation_failures = sum(
        row.get("evidence_source") == "attested_claim"
        and row.get("attestation_decision_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in row["attestation_decision_receipt"].items()
                if key != "decision_sha256"
            }
        )
        for row in records
    )
    result["artifacts"] = {
        "records_sha256": records_sha256,
        "records_reverified": sha256(records_payload).hexdigest() == records_sha256,
        "decision_receipt_failure_count": decision_failures,
        "attestation_decision_receipt_failure_count": attestation_failures,
        "conformance_sha256": sha256(_json_bytes(conformance)).hexdigest(),
    }
    results_payload = _json_bytes(result)
    report_payload = report_markdown(result).encode("utf-8")
    result_path = ROOT / "data" / "benchmarks" / f"{stem}_results.json"
    records_path = ROOT / "data" / "benchmarks" / f"{stem}_records.jsonl"
    report_path = ROOT / "reports" / f"{stem}.md"
    experiment_dir = ROOT / "experiments" / "attested_provenance_gate" / ("smoke" if args.smoke else "full")
    _write(result_path, results_payload)
    _write(records_path, records_payload)
    _write(report_path, report_payload)
    _write(experiment_dir / "results.json", results_payload)
    _write(experiment_dir / "records.jsonl", records_payload)
    _write(experiment_dir / "conformance.json", _json_bytes(conformance))
    _write(experiment_dir / "training_notes.md", report_payload)

    receipt = {
        "schema": "attested_provenance_gate_result_receipt_v1",
        "config_sha256": config_sha256,
        "results_path": str(result_path.relative_to(ROOT)),
        "results_sha256": sha256(results_payload).hexdigest(),
        "records_path": str(records_path.relative_to(ROOT)),
        "records_sha256": records_sha256,
        "records_reverified": sha256(records_path.read_bytes()).hexdigest() == records_sha256,
        "report_path": str(report_path.relative_to(ROOT)),
        "report_sha256": sha256(report_payload).hexdigest(),
        "conformance_path": str(conformance_path.relative_to(ROOT)),
        "conformance_sha256": sha256(conformance_path.read_bytes()).hexdigest(),
        "registry_receipts": result["registries"],
        "split_overlap": sum(row["group_overlap"] for row in result["split_receipts"]),
        "decision_receipt_failure_count": decision_failures,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    receipt_payload = _json_bytes(receipt)
    receipt_path = ROOT / "data" / "benchmarks" / f"{stem}_receipt.json"
    _write(receipt_path, receipt_payload)
    _write(experiment_dir / "result_receipt.json", receipt_payload)

    if not args.smoke:
        mirror = ROOT / "control_harness" / "attested_provenance_gate_v1"
        _write(mirror / "report.md", report_payload)
        _write(mirror / "result_receipt.json", receipt_payload)
        _write(mirror / "month1_summary.txt", (month1_summary(result) + "\n").encode("utf-8"))
    print(report_markdown(result))


if __name__ == "__main__":
    main()
