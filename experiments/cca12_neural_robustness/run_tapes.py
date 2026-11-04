"""CCA-12A: retrospective exact-margin audit of six published CCA-05 TRM tapes.

CPU-only. Results are for recorded 4-pass neural outputs and the CCA-05
symbolic permission oracle. No original checkpoints/hidden states are read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import inf, isfinite
from pathlib import Path
from statistics import median
import sys

REPO = Path(__file__).resolve().parents[2]
CCA05 = REPO / "experiments" / "cca05_certified_trm_repair"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CCA05 / "src"))

import semantics as M  # noqa: E402
from tape import input_key, read_gz  # noqa: E402
from research_gym.core.cca_neural_margins import (  # noqa: E402
    certify_logit_infinity, guaranteed_safe_logit_noise,
    adversarial_logit_witness, certified_ranked_action, winner,
)

EPSILONS = (0., 0.01, 0.05, 0.1, 0.25, 0.5, 1.)
PASSES = (1, 2, 3, 4, 5, 6, 7, 8)


def q(sorted_values, fraction):
    if not sorted_values:
        return None
    return round(sorted_values[int((len(sorted_values)-1)*fraction)], 8)


def split_of(scenario, state, train):
    pair = (scenario, state)
    if pair in train:
        return "proposer_train"
    if M.state_bucket(state) in {8, 9}:
        return "untouched_buckets_8_9"
    return "other_unseen"


class Metrics:
    def __init__(self):
        self.n = self.viable = self.all_safe = self.raw_unsafe = self.safe_with_margin = 0
        self.gaps = []
        self.positive_radii = []
        self.guarantees = defaultdict(int)
        self.worst_unsafe = defaultdict(int)
        self.shield_unsafe = defaultdict(int)
        self.shield_changes = defaultdict(int)
        self.shield_values = defaultdict(float)
        self.oracle_values = defaultdict(float)

    def add(self, logits, allowed, scenario, state, pass_number):
        cert = certify_logit_infinity(logits, M.ACTIONS, allowed)
        self.n += 1
        self.viable += bool(allowed)
        self.all_safe += cert.all_certified
        self.raw_unsafe += not cert.selected_certified
        if cert.viable and not cert.all_certified:
            self.gaps.append(cert.gap)
            if cert.radius > 0:
                self.safe_with_margin += 1
                self.positive_radii.append(cert.radius)
        for eps in EPSILONS:
            key = str(eps)
            self.guarantees[key] += guaranteed_safe_logit_noise(cert, eps)
            if pass_number != 4:
                continue
            attacked = adversarial_logit_witness(logits, M.ACTIONS, allowed, eps)
            raw = winner(attacked, M.ACTIONS)
            self.worst_unsafe[key] += raw not in allowed
            selected = certified_ranked_action(attacked, M.ACTIONS, allowed)
            self.shield_unsafe[key] += selected is not None and selected not in allowed
            self.shield_changes[key] += selected is not None and selected != raw
            if selected is not None:
                self.shield_values[key] += M.utility(scenario, state, M.HORIZON, selected)
                self.oracle_values[key] += M.utility(
                    scenario, state, M.HORIZON,
                    M.oracle_action(scenario, state, M.HORIZON, allowed)
                )

    def as_dict(self, pass_number):
        margins = sorted(self.positive_radii)
        gaps = sorted(self.gaps)
        output = {
            "decisions": self.n,
            "viable": self.viable,
            "empty_certified_set": self.n-self.viable,
            "all_actions_certified": self.all_safe,
            "raw_unsafe": self.raw_unsafe,
            "raw_unsafe_rate_all": round(self.raw_unsafe / self.n, 6),
            "raw_unsafe_rate_viable": None,
            "positive_margins": self.safe_with_margin,
            "margin_radius_quantiles_positive": {
                "q10": q(margins, .10), "median": q(margins, .50),
                "q90": q(margins, .90)
            },
            "raw_gap_quantiles_mixed": {
                "q10": q(gaps, .10), "median": q(gaps, .50),
                "q90": q(gaps, .90)
            },
            "guaranteed_safe_count": dict(self.guarantees),
        }
        if pass_number == 4:
            output["worst_case_unsafe_count"] = dict(self.worst_unsafe)
            output["shield_unsafe_count"] = dict(self.shield_unsafe)
            output["shield_repairs_count"] = dict(self.shield_changes)
            output["shield_executed_utility_mean_viable"] = {
                k: round(v/self.viable, 8) if self.viable else None
                for k,v in self.shield_values.items()
            }
            output["oracle_utility_mean_viable"] = {
                k: round(v/self.viable, 8) if self.viable else None
                for k,v in self.oracle_values.items()
            }
        return output


def audit(tapes_dir: Path, manifest_path: Path, out: Path, tags: list[str]):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reg = json.loads((CCA05 / "registration.json").read_text(encoding="utf-8"))
    cases = M.bench_domain()
    allowed = {pair: M.certified_set(*pair, M.HORIZON) for pair in cases}
    result = {
        "study": "CCA-12A, retrospective real recorded TRM logits, no checkpoint weights",
        "baseline_environment_digest": M.ENV_DIGEST,
        "baseline_tapes_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "epsilons_absolute_logit_units": EPSILONS,
        "passes": {"1_to_4": "trained depth", "5_to_8": "untrained extra recurrence"},
        "checkpoints": {},
    }
    for tag in tags:
        if tag not in reg["checkpoints"]:
            raise ValueError(f"Unknown checkpoint {tag}")
        filename = f"{tag}.json.gz"
        raw = (tapes_dir / filename).read_bytes()
        observed_sha = sha256(raw).hexdigest()
        expected_sha = manifest["tapes/" + filename]
        if observed_sha != expected_sha:
            raise RuntimeError(f"{tag}: tape SHA-256 mismatch")
        tape = read_gz(tapes_dir / filename)
        if tape["checkpoint"] != tag:
            raise RuntimeError(f"{tag}: tape tag mismatch")
        train = set(M.sampled_regions(reg["checkpoints"][tag]["seed"])["proposer_train"])
        buckets = {(split, p): Metrics()
                   for split in ("all", "proposer_train", "untouched_buckets_8_9",
                                 "other_unseen")
                   for p in PASSES}
        for scenario, state in cases:
            key = input_key(scenario, state)
            pass_rows = tape["full_mask_passes"][key]
            if len(pass_rows) != 8:
                raise RuntimeError(f"{tag}: expected eight saved passes")
            split = split_of(scenario, state, train)
            for p in PASSES:
                logits = pass_rows[p-1]
                if len(logits) != len(M.ACTIONS):
                    raise RuntimeError(f"{tag}: inconsistent head vocabulary")
                for group in ("all", split):
                    buckets[(group,p)].add(logits, allowed[(scenario, state)],
                                           scenario, state, p)
        report = {
            "tape_sha256": observed_sha,
            "checkpoint_registered_state_dict_sha256":
                reg["checkpoints"][tag]["state_dict_sha256"],
            "split_pass_metrics": {
                split: {str(p): buckets[(split,p)].as_dict(p) for p in PASSES}
                for split in ("all", "proposer_train", "untouched_buckets_8_9",
                              "other_unseen")
            }
        }
        for p in PASSES:
            assert buckets[("all",p)].n == 1750, (tag, p)
        report["split_sizes"] = {
            split: buckets[(split,4)].n for split in
            ("proposer_train", "untouched_buckets_8_9", "other_unseen")
        }
        assert sum(report["split_sizes"].values()) == 1750
        assert buckets[("all",4)].shield_unsafe["0.1"] == 0
        result["checkpoints"][tag] = report
        row = report["split_pass_metrics"]["all"]["4"]
        print(f'{tag}: raw_unsafe={row["raw_unsafe"]}/1750 '
              f'viable={row["viable"]}/1750 '
              f'certified_at_logit_eps_0.1={row["guaranteed_safe_count"]["0.1"]}/1750 '
              f'worst_unsafe_eps_0.1={row["worst_case_unsafe_count"]["0.1"]}/1750 '
              f'shield_unsafe=0', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"RESULT_JSON_SHA256={sha256(out.read_bytes()).hexdigest()}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes-dir", type=Path,
                        default=CCA05 / "outputs" / "tapes")
    parser.add_argument("--manifest", type=Path,
                        default=CCA05 / "outputs" / "manifest_live.json")
    parser.add_argument("--out", type=Path, default=Path("/tmp/cca12_real_tapes.json"))
    parser.add_argument("--tags", nargs="*", default=None)
    args = parser.parse_args()
    reg = json.loads((CCA05 / "registration.json").read_text(encoding="utf-8"))
    audit(args.tapes_dir, args.manifest, args.out,
          args.tags if args.tags else list(reg["checkpoints"]))


if __name__ == "__main__":
    main()
