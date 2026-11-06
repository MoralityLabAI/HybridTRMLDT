"""Post-gate Huginn factor interaction analysis. This never triggers an H100 run.

Reads outcomes generated on an untouched confirmation split, requiring a
preexisting PASS acceptance receipt from the calibration worker. One array per
pinned factor is required, along with all four fixed controls reused in both
windows. This module cannot independently certify the receipt's provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .control_repair import (clustered_bootstrap_ci, clustered_signflip_p,
                                 holm_rejections, window_effects)
except ImportError:
    from control_repair import (clustered_bootstrap_ci, clustered_signflip_p,
                                holm_rejections, window_effects)


# Frozen 8 hypotheses, in the original calibration registration.
PANEL = {
    "core1_factor19": "early_only",
    "core1_factor58": "late_only",
    "core1_factor26": "large_effect",
    "core1_factor31": "near_null",
    "core3_factor60": "early_only",
    "core3_factor17": "late_only",
    "core3_factor1": "large_effect",
    "core3_factor44": "near_null",
}


def evaluate(outcomes_path: Path, gate_receipt: Path, control_family: str,
             seed: int = 521, n_boot: int = 3000, n_perm: int = 9999) -> dict:
    gate = json.loads(gate_receipt.read_text())
    if gate.get("status") != "PASS" or gate.get("control_family") != control_family:
        raise ValueError("independently approved calibration PASS receipt required for this family")
    if set(gate.get("eligible_factors", [])) != set(PANEL):
        raise ValueError("all eight frozen factors must have eligible matched controls")
    if not gate.get("source_split_disjoint") or not gate.get("verification_frozen"):
        raise ValueError("seal and provenance assertions missing from acceptance receipt")
    results, pvals = {}, {}
    with np.load(outcomes_path, allow_pickle=False) as arrays:
        for idx, (factor_name, role) in enumerate(PANEL.items()):
            prefix = factor_name + "__" + control_family + "__"
            try:
                clean = arrays[prefix + "clean"]
                factor = arrays[prefix + "factor"]
                ctrls = arrays[prefix + "controls"]
            except KeyError as exc:
                raise ValueError(f"missing required source array: {exc}") from exc
            if clean.ndim != 3 or clean.shape[1] != 3 or clean.shape[-1] != 2:
                raise ValueError(f"{factor_name}: must have matched three-seed source-by-window arrays")
            effects = window_effects(clean, factor, ctrls)
            interaction = effects[:, :, 1] - effects[:, :, 0]
            stats = [clustered_bootstrap_ci(effects[:, :, window], seed + idx*101 + window, n_boot)
                     for window in range(2)]
            eq_stats = [clustered_bootstrap_ci(effects[:, :, window], seed + idx*101 + window, n_boot, confidence=.90)
                        for window in range(2)]
            interaction_ci = clustered_bootstrap_ci(interaction, seed + idx*101 + 3, n_boot)
            # Bonferroni simultaneous interval; Holm decision separately from sign-flip p.
            simultaneous_ci = clustered_bootstrap_ci(interaction, seed + idx*101 + 3, n_boot,
                                                      confidence=1-.05/len(PANEL))
            pvals[factor_name] = clustered_signflip_p(interaction, seed+idx*101, n_perm)
            result = {
                "role": role,
                "per_window_95pct_ci": {"early":stats[0], "late":stats[1]},
                "per_window_90pct_equivalence_ci": {"early":eq_stats[0], "late":eq_stats[1]},
                "interaction_95pct_ci": interaction_ci,
                "interaction_simultaneous_bonferroni_ci": simultaneous_ci,
                "interaction_signflip_p": pvals[factor_name],
                "n_clean_correct_programs": len(clean),
            }
            if role.endswith("_only"):
                active_idx = 0 if role == "early_only" else 1
                inactive_idx = 1 - active_idx
                result["active_harmful_and_95pct_excludes_zero"] = (
                    stats[active_idx]["mean"] < -.05 and stats[active_idx]["upper"] < 0)
                result["inactive_90pct_equivalent_to_zero"] = (
                    eq_stats[inactive_idx]["lower"] > -.05 and
                    eq_stats[inactive_idx]["upper"] < .05)
            results[factor_name] = result
    holm = holm_rejections(pvals)
    for k in PANEL:
        results[k]["interaction_holm_rejects_zero"] = holm[k]
    return {
        "status": "post_gate_analysis",
        "control_family": control_family,
        "n_hypotheses": len(PANEL),
        "paired_unit": "program_cluster_with_three_recurrent_noise_seeds",
        "results": results,
        "warning": "Statistical results remain conditional on independently verified GPU provenance, sealed data, and matching controls.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", required=True, type=Path)
    ap.add_argument("--gate", required=True, type=Path)
    ap.add_argument("--family", choices=("write_rotation", "read_rotation"), required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    result = evaluate(args.outcomes, args.gate, args.family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Analyzed {len(result['results'])} pre-registered factor interactions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
