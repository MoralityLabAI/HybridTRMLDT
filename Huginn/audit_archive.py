"""Read-only forensic audit of the Huginn closeout ZIP's raw calibration outputs.

Usage:
    python Huginn/audit_archive.py /path/to/unzipped/data --out /tmp/audit.json

It cannot infer a new HUG-CAL-02 result or compute covariance-aware controls from
saved norms alone: pre-projection hidden states and factors are absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    from .control_repair import matching_report
except ImportError:
    from control_repair import matching_report

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(data_dir: Path) -> dict:
    cal = json.loads((data_dir / "calibration.json").read_text())
    reg = json.loads((data_dir / "registration.json").read_text())
    report = json.loads((data_dir / "calibration_analysis.json").read_text())
    norms_path = data_dir / "calibration_norms.npz"
    expected_sha = cal["norms_sha256"]
    got_sha = sha256(norms_path)
    if got_sha != expected_sha:
        raise ValueError(f"calibration norms SHA256 differs: expected {expected_sha}, got {got_sha}")
    if cal["factors_sha256"] != reg["factor_sha256"]:
        raise ValueError("factor hash changed between calibration and registration")
    if cal["model_revision"] != reg["model_revision"]:
        raise ValueError("model revision changed between calibration and registration")
    if cal["programs_sha256"] != reg["generated_programs_sha256"]:
        raise ValueError("generated-program hash changed between calibration and registration")
    total = 0
    passed = 0
    full_pass = 0
    sample_count = 0
    by_kind = {}
    p10_only_factor = 0
    max_saved_quantile_error = 0.0
    feasible_oracle_cases = 0
    feasibility = []
    windows_same_controls = 0
    shared_control_pairs = 0
    with np.load(norms_path, allow_pickle=False) as norms:
        for key, val in sorted(cal["controls"].items()):
            parts = key.split("_")
            core, window = parts[0], parts[-1]
            kind = "factor" if "factor" in key else "svd" if "svd" in key else "remainder"
            slot = f"{kind}_{window}"
            by_kind.setdefault(slot, {"accepted": 0, "total": 0, "full_cases": 0})
            tq_key = f"{key}_verification_target"
            target = norms[tq_key]
            for j, selected in enumerate(val["selected"]):
                prefix = f"{key}_verification_control{j}"
                candidate = norms[prefix]
                if target.shape != (2880,) or candidate.shape != (2880,):
                    raise ValueError(f"unexpected norm tensor shape for {key}")
                observed = matching_report(target, candidate)
                expected = selected["verification"]
                if observed["accepted"] != expected["accepted"]:
                    raise ValueError(f"bad accepted flag: {key} control {j}")
                max_saved_quantile_error = max(max_saved_quantile_error,
                    float(np.max(np.abs(np.asarray(observed["candidate"]) -
                                        np.asarray(list(expected["candidate_quantiles"].values()))))))
                if not np.allclose(observed["candidate"], list(expected["candidate_quantiles"].values()),
                                   atol=3e-5, rtol=1e-5):
                    raise ValueError(f"quantile mismatch: {key} control {j}")
                if not np.allclose(observed["errors"], list(expected["relative_errors"].values()),
                                   atol=2e-5, rtol=1e-5):
                    raise ValueError(f"error mismatch: {key} control {j}")
                violations = [q for q,t in (("p10", .15), ("p50", .15), ("p90", .15), ("p95", .25))
                              if expected["relative_errors"][q] > t]
                if kind == "factor" and violations == ["p10"]:
                    p10_only_factor += 1
                total += 1
                passed += int(observed["accepted"])
                sample_count += candidate.size
                by_kind[slot]["accepted"] += int(observed["accepted"])
                by_kind[slot]["total"] += 1
            case_passed = sum(x["verification"]["accepted"] for x in val["selected"])
            full_pass += (case_passed == 4)
            by_kind[slot]["full_cases"] += (case_passed == 4)
            # This oracle uses already-read verification to assess candidate pool
            # feasibility, not to select or claim a result.
            feasible = sum(x["verification"]["accepted"] for x in val["candidates"])
            feasible_oracle_cases += feasible >= 4
            feasibility.append({"case": key, "selected_passed": case_passed,
                                "verification_pass_pool": feasible,
                                "pool_size": len(val["candidates"]),
                                "four_available_in_pool": feasible >= 4})
    for key in sorted(cal["controls"]):
        if not key.endswith("_early"):
            continue
        other = key.removesuffix("_early") + "_late"
        if other not in cal["controls"]:
            continue
        shared_control_pairs += 1
        a = [(item["seed"], item["index"], item.get("scale", 1.0))
             for item in cal["controls"][key]["selected"]]
        b = [(item["seed"], item["index"], item.get("scale", 1.0))
             for item in cal["controls"][other]["selected"]]
        windows_same_controls += (a == b)

    if total != 144 or len(cal["controls"]) != 36:
        raise ValueError(f"unexpected source size: {total} controls, {len(cal['controls'])} cases")
    if passed != cal["accepted_controls"] or passed != report["accepted_controls"]:
        raise ValueError("accepted-control count inconsistent with source reports")
    if full_pass != report["fully_matched_cases"]:
        raise ValueError("fully matched case count inconsistent with source report")
    return {
        "status": "retrospective_archive_audit_only",
        "provenance": {"model": reg["model"], "model_revision": reg["model_revision"],
                       "factor_sha256": reg["factor_sha256"],
                       "calibration_norms_sha256": got_sha,
                       "registration_generated_programs_sha256": reg["generated_programs_sha256"],
                       "calibration_sha256": sha256(data_dir / "calibration.json")},
        "recomputed": {"selected_controls_passed": passed, "selected_controls_total": total,
                       "fully_matched_cases": full_pass, "total_cases": len(cal["controls"]),
                       "candidate_pool_cases_with_four_or_more_oracle_feasible": feasible_oracle_cases,
                       "candidate_pool_cases_with_fewer_than_four": len(cal["controls"]) - feasible_oracle_cases,
                       "p10_only_failures_among_selected_factor_controls": p10_only_factor,
                       "window_pairs_reusing_same_four_control_ids": windows_same_controls,
                       "window_pairs_total": shared_control_pairs,
                       "selected_verification_norm_samples": sample_count,
                       "max_abs_saved_candidate_quantile_discrepancy": max_saved_quantile_error},
        "by_kind": by_kind,
        "candidate_feasibility_posthoc": feasibility,
        "warnings": [
            "Oracle feasibility inspects verification labels and cannot authorize new confirmation.",
            "Earlier early/late control pools use different seeds and sampled directions. A new temporal estimand should reuse one matrix across windows.",
            "Saved per-control norms cannot reproduce input covariance or causally test factor ablations.",
            "The old calibration verification partition has been inspected; use a fresh calibration split before prospective acceptance.",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="unzipped closeout data/ directory")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    result = audit(args.data_dir)
    raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(raw)
    print(json.dumps(result["recomputed"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
