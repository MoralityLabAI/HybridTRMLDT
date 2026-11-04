"""Dual-LDT-03: frozen-world manifest and paired prospective synthetic replay."""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.dual_ldt_01.src.run import CHARACTERS, load_heads  # noqa: E402
from experiments.dual_ldt_03.src.world import (  # noqa: E402
    NAMESPACE, all_worlds, assert_game_model,
)
from experiments.dual_ldt_03.src.episode import (  # noqa: E402
    evaluate_episode, verify_preference_correction,
)


def canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(obj)
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def manifest():
    worlds = [w.to_jsonable() for w in all_worlds()]
    return {
        "status": "frozen_config_not_an_outcome",
        "namespace": NAMESPACE,
        "world_configuration_sha256": sha256(canonical(worlds)).hexdigest(),
        "development_ids": [w["world_id"] for w in worlds if w["split"]=="development"],
        "report_ids": [w["world_id"] for w in worlds if w["split"]=="report"],
        "worlds": worlds,
    }


def summary(episodes):
    by_group = defaultdict(list)
    lookup = {}
    for e in episodes:
        by_group[(e["split"], e["disturbance"], e["persona"], e["policy"])].append(e)
        key = (e["world"], e["persona"], e["disturbance"], e["policy"])
        if key in lookup:
            raise ValueError("duplicate episode key")
        lookup[key] = e
    rows = []
    for (split, disturbance, persona, policy), group in sorted(by_group.items()):
        rows.append({
            "split": split, "disturbance": disturbance,
            "persona": persona, "policy": policy, "n": len(group),
            "halt_success": sum(e["host_halt"] for e in group),
            "halt_rate": mean(float(e["host_halt"]) for e in group),
            "mean_reward": mean(e["task_reward"] for e in group),
            "mean_preference_loss": mean(e["preference_loss"] for e in group),
            "mean_added_tax": mean(e["additional_preference_tax"] for e in group),
            "mean_decisions": mean(e["action_count"] for e in group),
            "mean_host_moves": mean(e["host_moves"] for e in group),
            "mean_changed_rank": mean(e["counterfactual_choice_changes"] for e in group),
        })
    paired_by_group = defaultdict(list)
    unique = sorted({(e["world"],e["persona"],e["disturbance"]) for e in episodes})
    for world,persona,disturbance in unique:
        immediate = lookup[(world,persona,disturbance,"immediate")]
        temporal = lookup[(world,persona,disturbance,"temporal")]
        first_i = next((r.get("action") for r in immediate["steps"] if "action" in r),None)
        first_t = next((r.get("action") for r in temporal["steps"] if "action" in r),None)
        paired_by_group[(immediate["split"],disturbance,persona)].append({
            "halt_delta": int(temporal["host_halt"])-int(immediate["host_halt"]),
            "reward_delta": temporal["task_reward"]-immediate["task_reward"],
            "first_choice_changed": first_i != first_t,
            "persona_loss_delta": temporal["preference_loss"]-immediate["preference_loss"],
            "additional_tax": temporal["additional_preference_tax"],
        })
    paired = []
    for (split, disturbance, persona), group in sorted(paired_by_group.items()):
        paired.append({
            "split": split, "disturbance": disturbance, "persona": persona,
            "n_worlds": len(group),
            "delta_halt_rate": mean(v["halt_delta"] for v in group),
            "delta_reward": mean(v["reward_delta"] for v in group),
            "delta_persona_loss": mean(v["persona_loss_delta"] for v in group),
            "mean_extra_tax": mean(v["additional_tax"] for v in group),
            "first_choice_changed_rate": mean(float(v["first_choice_changed"]) for v in group),
        })
    return {"grouped": rows, "paired": paired}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("experiments/dual_ldt_03/outputs"))
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--split", choices=["all","development","report"], default="all")
    p.add_argument("--limit", type=int, default=0,
                   help="deterministic prefix smoke; 0 is the full selected split")
    args = p.parse_args()
    if args.limit < 0:
        p.error("limit must be nonnegative")
    receipt = manifest()
    manifest_sha = write(args.out/"manifest.json", receipt)
    print("DUAL-LDT-03 MANIFEST", manifest_sha,
          "worlds", len(receipt["worlds"]))
    if args.manifest_only:
        return
    heads, head_hashes = load_heads()
    worlds = [w for w in all_worlds()
              if args.split=="all" or args.split==w.split]
    if args.limit:
        worlds = worlds[:args.limit]
    for world in worlds:
        assert_game_model(world)
        for head in heads.values():
            verify_preference_correction(world,head)
    episodes = [
        evaluate_episode(world, persona, heads[persona], policy, disturbance)
        for world in worlds
        for persona in CHARACTERS
        for disturbance in ("worst","benign")
        for policy in ("immediate","temporal")
    ]
    if any(e["policy"]=="temporal" and not e["host_halt"] for e in episodes):
        raise AssertionError("temporal controller broke human intervention")
    aggregate = summary(episodes)
    result = {
        "status": "synthetic_prospectively_frozen_environment_diagnostic",
        "generator_commit": "c521526259c8f1245d49d9dea22d088dbed64691",
        "registration_commit": "710144b226d0c514c1c53ce39c8671211a551e5a",
        "manifest_sha256": manifest_sha,
        "world_configuration_sha256": receipt["world_configuration_sha256"],
        "heads": head_hashes,
        "world_ids": [w.world_id for w in worlds],
        "episodes": episodes,
    }
    raw_sha = write(args.out/"episodes.json", result)
    summarized = {
        "status": result["status"], "split": args.split,
        "episode_count": len(episodes), "world_count": len(worlds),
        "manifest_sha256": manifest_sha,
        "episodes_sha256": raw_sha,
        **aggregate,
    }
    write(args.out/"summary.json", summarized)
    for r in summarized["paired"]:
        print("DUAL-LDT-03",r["split"],r["disturbance"],r["persona"],
              "HALTΔ",round(r["delta_halt_rate"],4),
              "rewardΔ",round(r["delta_reward"],4),
              "first-choiceΔ",round(r["first_choice_changed_rate"],4))
    print("WROTE",args.out/"summary.json",args.out/"episodes.json")


if __name__=="__main__":
    main()
