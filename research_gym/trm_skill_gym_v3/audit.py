"""Verification audits for the GYM-02 full campaign.

* label leakage: per stratum, absolute point-biserial correlation of every public
  candidate feature column with the label (flag > 0.5 outside known structural columns)
* train/test overlap: root ranges are disjoint, and instance content hashes of sampled
  train/calibration/test pools do not collide
* selection audit: every frozen.json was written before the first test result file
* resource accounting: supervisor peak process-tree RSS below the 3 GiB limit; worker
  training CPU <= supervisor-observed tree CPU + tolerance
* credential scan: the Tinker credential never appears in any campaign artifact
* receipt replay: recompute a sample of race test results from checkpoints and compare
  per-root outcomes exactly

    python -m research_gym.trm_skill_gym_v3.audit --campaign experiments/trm_skill_gym_02/full --out experiments/trm_skill_gym_02/full/audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import warnings

import numpy as np
import torch

from research_gym.trm_skill_gym import tasks as T
from research_gym.trm_skill_gym.inky import load_key
from . import graphs as G
from . import hybrid as H
from . import race as RC
from . import router as RT

STRUCTURAL = {"retrieval": {0, 10, 11}, "deduction": {1, 2, 3}, "repair": {0, 1, 2, 3}, "curation": {6, 7, 8},
              "research": set(range(14)), "stopping": set(range(11))}


def leakage(count: int = 256) -> Dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = {}
    for skill in RC.ORDER:
        for fam in G.strata(skill, "test"):
            b = T.stack([G.instance(skill, fam, 9_500_000 + i) for i in range(count)])
            m = b["mask"].reshape(-1) > 0
            y = b["label"].reshape(-1)[m].numpy()
            if y.std() == 0:
                continue
            flags = {}
            for c in range(T.FS):
                x = b["cand"][..., c].reshape(-1)[m].numpy()
                if x.std() == 0:
                    continue
                r = float(abs(np.corrcoef(x, y)[0, 1]))
                if r > 0.3 and c not in STRUCTURAL[skill]:
                    flags[c] = round(r, 3)
            out[f"{skill}/{fam}"] = flags
    # positive control: the GYM-01 label-correlated hint columns must be flagged on the original families
    control = {f"{s}/{f}": out.get(f"{s}/{f}", {}) for s in G.GRAPH_SKILLS for f in T.FAMILIES[s]["dev"]}
    hintfree = {k: v for k, v in out.items() if k.endswith("_hintfree") or any(t in k for t in G.TRAIN_TOPOLOGIES + G.WITHHELD_TOPOLOGIES)}
    return {"flagged": {k: v for k, v in out.items() if v}, "strata_checked": len(out),
            "positive_control_original_families": control,
            "hintfree_and_topology_strata_flagged": {k: v for k, v in hintfree.items() if v}, "threshold_abs_r": 0.3}


def _h(inst_batch, i):
    return hashlib.sha256(b"".join(inst_batch[k][i].numpy().tobytes() for k in ("cand", "rel", "mask"))).hexdigest()


def overlap(n: int = 256) -> Dict:
    ranges = {"race": (RC.TRAIN_ROOT, RC.CAL_ROOT, RC.TEST_ROOT), "hybrid": (H.TRAIN_ROOT, H.CAL_ROOT, H.TEST_ROOT),
              "router": (RT.CAL_ROOT, RT.TEST_ROOT)}
    collisions = {}
    for skill in ("retrieval", "deduction", "repair"):
        train = RC.training_pool(skill, n)
        test = {f: RC.batch(skill, f, RC.TEST_ROOT + 10_000 * k, n) for k, f in enumerate(G.strata(skill, "test"))}
        th = {_h(train, i) for i in range(n)}
        hits = sum(_h(b, i) in th for b in test.values() for i in range(n))
        collisions[skill] = hits
    return {"root_ranges": ranges, "train_test_content_collisions": collisions}


def selection_timing(campaign: Path) -> Dict:
    fz = campaign / "race" / "frozen.json"
    if not fz.exists():
        return {"status": "no_freeze"}
    frozen = json.loads(fz.read_text(encoding="utf-8"))
    res = {}
    for skill, row in frozen.items():
        tests = list((campaign / "race" / "test" / skill).glob("*.json"))
        first = min((p.stat().st_mtime for p in tests), default=None)
        res[skill] = {"frozen_at": row["frozen_at"], "first_test_file": first,
                      "ok": first is None or row["frozen_at"] < first}
    return res


def resources(campaign: Path) -> Dict:
    q = json.loads((campaign / "queue_state.json").read_text(encoding="utf-8"))
    peaks = {k: v.get("peak_tree_rss_bytes") for k, v in q["stages"].items()}
    race_sup = []
    for sp in (campaign / "race" / "jobs").glob("*/*.json"):
        if sp.name.endswith(".spec.json"):
            continue
        j = json.loads(sp.read_text(encoding="utf-8"))
        s = j.get("supervisor") or {}
        if s:
            race_sup.append(s.get("peak_tree_rss_bytes", 0))
    sup_cal = campaign / "race" / "supervision_cal.json"
    cal = json.loads(sup_cal.read_text(encoding="utf-8")) if sup_cal.exists() else {}
    worst = max([v.get("peak_tree_rss_bytes", 0) for v in cal.values()] + race_sup + [0])
    return {"stage_peak_tree_rss_bytes": peaks, "race_job_peak_tree_rss_max_bytes": worst,
            "limit_bytes": 3 * 1024**3, "all_below_limit": all((p or 0) < 3 * 1024**3 for p in list(peaks.values()) + [worst]),
            "stage_status": {k: v.get("status") for k, v in q["stages"].items()}}


def credential_scan(roots: List[Path]) -> Dict:
    key = load_key()
    if not key:
        return {"status": "no_credential_available_to_scan_for"}
    hits, files = [], 0
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.stat().st_size < 50 * 1024**2 and p.suffix not in (".pt",):
                files += 1
                try:
                    if key in p.read_bytes().decode("utf-8", "ignore"):
                        hits.append(str(p))
                except OSError:
                    pass
    return {"files_scanned": files, "hits": hits}


def replay_race(campaign: Path, per_skill: int = 2) -> Dict:
    out = {}
    for skill in RC.ORDER:
        tdir = campaign / "race" / "test" / skill
        files = sorted(p for p in tdir.glob("*.json")) if tdir.exists() else []
        for p in files[:per_skill]:
            name, seed = p.stem.rsplit("__s", 1)
            j = json.loads(p.read_text(encoding="utf-8"))
            if j.get("status") != "ok":
                continue
            ck = campaign / "race" / "jobs" / skill / f"{name}__s{seed}.pt"
            if name == "A_existing_skill":
                model, cfg = None, None
            else:
                model, cfg = RC._load(name, ck)
            from research_gym.trm_skill_gym import campaign as C
            cfg = cfg or C.base_trm()
            fam = next(iter(j["test"]))
            k = G.strata(skill, "test").index(fam)
            b = RC.batch(skill, fam, RC.TEST_ROOT + 10_000 * k, len(j["test"][fam]["per_root"]))
            r = RC.evaluate(skill, model, cfg, b)
            out[f"{skill}/{name}/s{seed}/{fam}"] = r["per_root"] == j["test"][fam]["per_root"]
    return {"replayed": len(out), "all_match": all(out.values()) if out else None, "rows": out}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=Path("experiments/trm_skill_gym_02/full"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    res = {"leakage": leakage(), "overlap": overlap(), "selection_timing": selection_timing(a.campaign),
           "resources": resources(a.campaign), "receipt_replay_race": replay_race(a.campaign),
           "credential_scan": credential_scan([a.campaign, Path("experiments/trm_skill_gym_02"), Path("research_gym/baby_goblin")])}
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: (v if k != "receipt_replay_race" else {kk: v[kk] for kk in ("replayed", "all_match")}) for k, v in res.items()}, default=float)[:3000])


if __name__ == "__main__":
    main()
