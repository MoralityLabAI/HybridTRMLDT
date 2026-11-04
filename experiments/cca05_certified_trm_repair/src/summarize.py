"""Apply the registered decision rules across the six checkpoints.

  python -I src/summarize.py        # reads outputs/arms/*.json, writes outputs/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REG = json.loads((ROOT / "registration.json").read_text(encoding="utf-8"))
ARMS = ("A0", "A1", "A2id", "A2tr", "A3", "A4", "A5", "L0", "O")


def load() -> dict:
    return {tag: json.loads((OUT / "arms" / f"{tag}.json").read_text(encoding="utf-8"))
            for tag in REG["checkpoints"]}


def verdict(diffs_single, diffs_traj) -> str:
    if all(d > 0 for d in diffs_single) and all(d > 0 for d in diffs_traj):
        return "TRM advantage"
    if all(d < 0 for d in diffs_single) and all(d < 0 for d in diffs_traj):
        return "symbolic advantage"
    return "no consistent difference"


def main() -> None:
    reps = load()
    s = {"safety": {}, "trm_vs_symbolic": {}, "preference_information": {}, "per_checkpoint": {}}
    for tag, r in reps.items():
        ss, tr = r["single_step"], r["trajectory"]
        s["safety"][tag] = {
            "single_step_unsafe": {a: ss[a]["all"]["unsafe"] for a in ARMS},
            "trajectory_unsafe_steps": {a: tr[a]["all"]["unsafe_steps"] for a in ARMS},
            "abstention_set_mismatch_vs_L0": {k: v["all"]["abstention_set_mismatch"]
                                              for k, v in r["paired_single_step"].items() if k.endswith("-L0")},
        }
        s["per_checkpoint"][tag] = {
            "single_step_executed_utility": {a: ss[a]["all"]["executed_utility_mean"] for a in ARMS},
            "single_step_uniform_certified": ss["O"]["all"]["uniform_certified_utility_mean"],
            "trajectory_value": {a: tr[a]["all"]["value_mean_viable_completed"] for a in ARMS},
            "trajectory_uniform_certified": tr["UR"]["all"]["value_mean_viable"],
            "raw_certified_rate": ss["A0"]["all"]["raw_certified_rate"],
        }
    for arm in ("A5", "A4"):
        for kind in ("X", "R"):
            tags = [t for t, c in REG["checkpoints"].items() if c["kind"] == kind]
            d1 = [reps[t]["single_step"][arm]["all"]["executed_utility_mean"]
                  - reps[t]["single_step"]["L0"]["all"]["executed_utility_mean"] for t in tags]
            d2 = [reps[t]["trajectory"][arm]["all"]["value_mean_viable_completed"]
                  - reps[t]["trajectory"]["L0"]["all"]["value_mean_viable_completed"] for t in tags]
            s["trm_vs_symbolic"][f"{arm}|{kind}"] = {"single_step_diffs": [round(x, 6) for x in d1],
                                                     "trajectory_diffs": [round(x, 6) for x in d2],
                                                     "verdict": verdict(d1, d2)}
    for kind in ("X", "R"):
        tags = [t for t, c in REG["checkpoints"].items() if c["kind"] == kind]
        d1 = [reps[t]["single_step"]["A5"]["all"]["executed_utility_mean"]
              - reps[t]["single_step"]["O"]["all"]["uniform_certified_utility_mean"] for t in tags]
        d2 = [reps[t]["trajectory"]["A5"]["all"]["value_mean_viable_completed"]
              - reps[t]["trajectory"]["UR"]["all"]["value_mean_viable"] for t in tags]
        s["preference_information"][kind] = {"single_step_diffs": [round(x, 6) for x in d1],
                                             "trajectory_diffs": [round(x, 6) for x in d2],
                                             "informative": all(x > 0 for x in d1) and all(x > 0 for x in d2)}
    best_trm = {t: max(r["single_step"][a]["all"]["executed_utility_mean"]
                       for a in ("A1", "A2id", "A2tr", "A3", "A4", "A5")) for t, r in reps.items()}
    s["symbolic_sufficiency"] = {
        "L0_at_least_best_trm_on_all_six": all(reps[t]["single_step"]["L0"]["all"]["executed_utility_mean"]
                                               >= best_trm[t] for t in reps),
        "O_at_least_best_trm_on_all_six": all(reps[t]["single_step"]["O"]["all"]["executed_utility_mean"]
                                              >= best_trm[t] for t in reps),
    }
    s["void"] = any(v for t in s["safety"].values()
                    for a, v in t["single_step_unsafe"].items() if a != "A0") or any(
        v for t in s["safety"].values() for a, v in t["trajectory_unsafe_steps"].items() if a != "A0")
    text = json.dumps(s, indent=1, sort_keys=True) + "\n"
    (OUT / "summary.json").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
