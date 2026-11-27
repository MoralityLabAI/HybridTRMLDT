"""Aggregate a TRM-SKILL-GYM-01 run: decision rules, skill packages, GoblinBench
post-training records and REPORT.md.

    python -m research_gym.trm_skill_gym.report --run experiments/trm_skill_gym_01/run --out experiments/trm_skill_gym_01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import torch

from . import dsl
from . import modules as M
from . import skill_templates as S
from . import tasks as T
from . import train as TR

ARMS = ["A_existing_skill", "B_feedforward", "C_gru", "D_trm_direct", "E_trm_matched_damping", "F_trm_rlt",
        "G_trm_ldt", "H_inky_engineered"]
MODULE_ARMS = ARMS[3:]
MARGIN = 0.02


def rj(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n") if Path(p).suffix in (".md", ".json", ".py", ".jsonl")
                          else Path(p).read_bytes()).hexdigest()


def wj(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8", newline="\n")


def mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def load_jobs(run: Path, kind: str) -> List[Dict]:
    d = run / kind
    return [rj(p) for p in sorted(d.glob("*.json"))] if d.exists() else []


def aggregate_main(jobs: List[Dict]) -> Dict:
    out: Dict[str, Dict[str, Dict]] = {}
    for skill in T.SKILLS:
        out[skill] = {}
        for arm in ARMS:
            js = [j for j in jobs if j["skill"] == skill and j["arm"] == arm and j.get("status") == "ok"]
            if not js:
                continue
            g = lambda pool, key: mean([j.get(pool, {}).get(key) for j in js])
            row = {
                "seeds": sorted(j["seed"] for j in js),
                "implementation": js[0]["implementation"],
                "params": js[0]["params"],
                "dev_exact": g("frozen_dev", "exact"),
                "withheld_exact": g("frozen_withheld", "exact"),
                "dev_exact_per_seed": [j["frozen_dev"]["exact"] for j in js],
                "dev_repair_success": g("frozen_dev", "repair_success"),
                "withheld_repair_success": g("frozen_withheld", "repair_success"),
                "dev_ece": g("frozen_dev", "ece"),
                "dev_adaptive_exact": g("frozen_dev", "adaptive_exact"),
                "dev_adaptive_visits": g("frozen_dev", "adaptive_visits"),
                "dev_adaptive_verifier_calls": g("frozen_dev", "adaptive_verifier_calls"),
                "verifier_calls": g("frozen_dev", "verifier_calls_per_instance"),
                "flops_per_instance": g("frozen_dev", "flops_per_instance"),
                "dev_exact_visit1_pre_verifier": mean([j["frozen_dev"]["exact_by_visit"][0] for j in js]),
                "train_seconds": mean([j["train"].get("train_seconds") for j in js]),
                "peak_rss_delta_mib": mean([j["train"].get("peak_rss_delta_bytes", 0) / 2**20 for j in js]),
                "dev_exact_1visit": g("frozen_dev_1", "exact"),
                "dev_exact_2R": g("frozen_dev_2R", "exact"),
                "withheld_exact_1visit": g("frozen_withheld_1", "exact"),
                "withheld_exact_2R": g("frozen_withheld_2R", "exact"),
                "certificate_violations": sum(j.get("certificate_violations", 0) for j in js),
                "config": js[0]["config"],
            }
            out[skill][arm] = row
    return out


def decisions(main: Dict, ablations: List[Dict]) -> Dict:
    dec = {}
    for skill, rows in main.items():
        d = {}
        base = [rows[a]["dev_exact"] for a in ("A_existing_skill", "B_feedforward", "C_gru") if a in rows]
        mods = {a: rows[a] for a in MODULE_ARMS if a in rows}
        if base and mods:
            best_arm = max(mods, key=lambda a: mods[a]["dev_exact"])
            b = mods[best_arm]
            a_with = rows.get("A_existing_skill", {}).get("withheld_exact")
            promote = (b["dev_exact"] >= max(base) + MARGIN and (a_with is None or b["withheld_exact"] >= a_with)
                       and b["certificate_violations"] == 0)
            d["best_module_arm"] = best_arm
            d["best_module_dev_exact"] = b["dev_exact"]
            d["baseline_max_dev_exact"] = max(base)
            d["promoted"] = bool(promote)
        if all(a in rows for a in ("B_feedforward", "C_gru")):
            px = max(rows["B_feedforward"]["dev_exact"], rows["C_gru"]["dev_exact"])
            trm = max(rows.get(a, {}).get("dev_exact", -1) for a in ("D_trm_direct", "F_trm_rlt"))
            d["trm_beats_proxies"] = bool(trm >= px + MARGIN)
            d["trm_minus_best_proxy"] = trm - px
        if "D_trm_direct" in rows and "H_inky_engineered" in rows:
            d["H_minus_D"] = rows["H_inky_engineered"]["dev_exact"] - rows["D_trm_direct"]["dev_exact"]
        if "D_trm_direct" in rows and rows["D_trm_direct"]["dev_exact_1visit"] is not None:
            d["recurrence_gain_D"] = rows["D_trm_direct"]["dev_exact"] - rows["D_trm_direct"]["dev_exact_1visit"]
            d["recurrence_matters_D"] = bool(d["recurrence_gain_D"] >= MARGIN)
        sh = [j for j in ablations if j["skill"] == skill and j["arm"] == "D_shuffled_supervision" and j.get("status") == "ok"]
        if sh and "D_trm_direct" in rows:
            se = sh[0]["frozen_dev"]["exact"]
            d["shuffled_dev_exact"] = se
            d["shuffled_near_chance"] = bool(se <= max(0.05, 0.5 * rows["D_trm_direct"]["dev_exact"]))
        for name in ("D_latent_reset", "F_latent_reset"):
            ab = [j for j in ablations if j["skill"] == skill and j["arm"] == name and j.get("status") == "ok"]
            ref = "D_trm_direct" if name.startswith("D") else "F_trm_rlt"
            if ab and ref in rows:
                d[f"{name}_minus_{ref}"] = ab[0]["frozen_dev"]["exact"] - rows[ref]["dev_exact"]
        dec[skill] = d
    n_h = sum(1 for d in dec.values() if d.get("H_minus_D") is not None and d["H_minus_D"] > 0)
    dec["_global"] = {"skills_where_H_beats_D": n_h, "inky_adds_value_rule_H_vs_D": n_h >= 4,
                      "skills_promoted": [s for s, d in dec.items() if d.get("promoted")],
                      "skills_trm_beats_proxies": [s for s, d in dec.items() if d.get("trm_beats_proxies")]}
    return dec


def search_summary(state: Dict) -> Dict:
    out = {}
    for arm in ("meta_direct", "meta_rlt", "random_mutation"):
        es = [e for e in state.get("archive", []) if e["arm"] == arm]
        pred = [(e["predicted_dev_exact"], e["measured_dev_exact"]) for e in es
                if isinstance(e.get("predicted_dev_exact"), (int, float)) and e.get("measured_dev_exact") is not None]
        out[arm] = {
            "rounds": state.get("round", {}).get(arm),
            "proposals": len(es),
            "schema_rejected": sum(e.get("status") == "schema_rejected" for e in es),
            "no_change": sum(e.get("status") == "no_change" for e in es),
            "trained": sum(e.get("status") == "ok" for e in es),
            "promoted_to_incumbent": sum(e.get("decision") == "promoted_to_incumbent" for e in es),
            "prediction_mae": mean([abs(p - m) for p, m in pred]),
            "prediction_bias": mean([p - m for p, m in pred]),
            "n_predictions": len(pred),
        }
    return out


@torch.no_grad()
def paired_textonly(run: Path, main_jobs: List[Dict]) -> Dict:
    out = {}
    for skill in T.SKILLS:
        p = run / "text_only" / f"{skill}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        recs = [r for r in recs if r.get("status") == "ok"]
        if not recs:
            continue
        n = max(r["index"] for r in recs) + 1
        batch = T.make_batch(skill, T.FAMILIES[skill]["dev"], "frozen_eval", 0, n)
        idx = [r["index"] for r in recs]
        row = {"n": len(recs), "text_only_exact": mean([r["exact"] for r in recs]),
               "text_only_parsed": mean([float(r["parsed"]) for r in recs]),
               "text_only_completion_tokens": mean([(r.get("usage") or {}).get("completion_tokens") for r in recs])}
        heur = M.episode(None, skill, batch, dict(dsl.defaults(), ldt_projection=False), heuristic=True)
        row["A_visit1_exact"] = mean(heur["exact"][0][idx].tolist())
        row["A_final_exact"] = mean(heur["exact"][-1][idx].tolist())
        for arm in MODULE_ARMS + ["B_feedforward", "C_gru"]:
            js = [j for j in main_jobs if j["skill"] == skill and j["arm"] == arm and j["seed"] == 0 and j.get("checkpoint")]
            if not js:
                continue
            ck = torch.load(run / js[0]["checkpoint"])
            model = M.SkillModule(ck["config"])
            model.load_state_dict(ck["state_dict"])
            model.eval()
            o = M.episode(model, skill, batch, ck["config"])
            row[f"{arm}_visit1_exact"] = mean(o["exact"][0][idx].tolist())
            row[f"{arm}_final_exact"] = mean(o["exact"][-1][idx].tolist())
        out[skill] = row
    return out


def export_packages(run: Path, out: Path, main: Dict, dec: Dict, frozen: Dict, receipts: Dict) -> Dict:
    pk = {}
    for skill, d in dec.items():
        if skill.startswith("_") or not d.get("promoted"):
            continue
        arm = d["best_module_arm"]
        row = main[skill][arm]
        ck = run / "checkpoints" / f"{skill}__{arm}__s0.pt"
        if not ck.exists():
            continue
        dst = out / "packages" / f"{skill}_{arm}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ck, dst / "weights.pt")
        cfg = row["config"]
        wj(dst / "module_config.json", {"skill": skill, "arm": arm, "implementation": row["implementation"],
                                        "config": cfg, "config_id": dsl.config_id(cfg), "params": row["params"]})
        wj(dst / "observation_schema.json", {
            "candidates": T.N, "candidate_features": T.FC, "skill_feature_columns": T.FS, "feedback_columns": T.FB,
            "feedback_layout": ["verified", "ok", "diag0", "diag1"], "ldt_certificate_column": M.CERT_COL,
            "global_features": T.G, "relation": S.RELATION[skill], "named_features": S.FEATURES[skill],
            "label_kind": T.LABEL_KIND[skill]})
        wj(dst / "verifier_contract.json", {
            "verifier": f"research_gym.trm_skill_gym.tasks.verify(skill='{skill}')",
            "returns": {"ok": "bool receipt verdict", "diag": "[N, 2] structured per-candidate diagnostics"},
            "in_loop_calls": cfg["verifier_visits"], "final_receipt_calls": 1,
            "ldt_certifier": "research_gym.trm_skill_gym.tasks.certify" if cfg["ldt_projection"] else None,
            "module_never_issues_receipts": True})
        md = S.render(skill) + f"""
## Trained module

- Implementation: `{row['implementation']}` ({row['params']} parameters), config `{dsl.config_id(cfg)}` in
  `module_config.json`, weights in `weights.pt`.
- Protocol: {cfg['visits']} visits, {cfg['verifier_visits']} in-loop verifier call(s) plus the final receipt; halting at
  confidence >= 0.85 or predicted value of another recurrence <= 0.
- Frozen evaluation (fresh roots, mean over seeds {row['seeds']}): dev exact {f3(row['dev_exact'])}, withheld-family exact
  {f3(row['withheld_exact'])}; strongest baseline (A/B/C) {f3(d['baseline_max_dev_exact'])}.
"""
        (dst / "SKILL.md").write_text(md, encoding="utf-8", newline="\n")
        ev = {"skill": skill, "arm": arm, "metrics": {k: v for k, v in row.items() if k != "config"},
              "decision": d, "frozen_architectures_sha256": frozen.get("sha256"),
              "source_receipts": receipts, "weights_sha256": sha_file(dst / "weights.pt"),
              "module_config_sha256": sha_file(dst / "module_config.json")}
        wj(dst / "evaluation_receipts.json", ev)
        pk[skill] = str(dst.relative_to(out))
    return pk


def goblinbench_records(run: Path, out: Path, state: Dict, frozen: Dict, dec: Dict, main: Dict) -> int:
    archive = state.get("archive", [])
    split_hash = hashlib.sha256(json.dumps(T.ROOT_RANGES, sort_keys=True).encode()).hexdigest()
    manifest = hashlib.sha256(json.dumps({"families": T.FAMILIES, "roots": T.ROOT_RANGES}, sort_keys=True).encode()).hexdigest()
    n = 0
    path = out / "goblinbench_records.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, e in enumerate(archive):
            later = [x for x in archive[i + 1:] if x["arm"] == e["arm"] and x["skill"] == e["skill"] and x.get("config")]
            revised = later[0]["config"] if later else None
            fz = frozen.get("architectures", {}).get(e["skill"], {})
            final = e.get("decision")
            if e.get("config") is not None and fz.get("config") == e.get("config") and e["arm"] in ("meta_direct", "meta_rlt"):
                d = dec.get(e["skill"], {})
                final = f"frozen_as_H; package_promoted={bool(d.get('promoted') and d.get('best_module_arm') == 'H_inky_engineered')}"
                hrow = main.get(e["skill"], {}).get("H_inky_engineered")
            else:
                hrow = None
            res = e.get("rung1") or e.get("rung0") or {}
            rec = {
                # TrainingCandidate v1 compatibility (DataGoblin schemas/training_candidate.schema.json)
                "schema_version": "1.0",
                "candidate_id": f"trm-skill-gym-01/{e['arm']}/r{e['round']}/p{e['proposal_index']}/{e.get('config_id', 'rejected')}",
                "architecture_family": dsl.implementation_label(e["config"]) if e.get("config") else "schema_rejected",
                "mock": False,
                "weights_loaded": False,
                "training_splits": ["train"],
                "training_split_hashes": {"train": split_hash},
                "dataset_manifest_sha256": manifest,
                "maximum_loops": (e.get("config") or {}).get("visits", 0) or 1,
                "resource_summary": {"params": e.get("params"), "train_seconds": res.get("train_seconds"),
                                     "peak_rss_delta_bytes": res.get("peak_rss_delta_bytes"),
                                     "flops_per_instance": res.get("flops_per_instance"),
                                     "verifier_calls_per_instance": res.get("verifier_calls"),
                                     "visits": res.get("visits"), "inky": e.get("inky"),
                                     "round_seconds": e.get("round_seconds")},
                # GoblinBench post-training fields
                "record_schema": "goblinbench.trm_skill_gym.post_training.v1",
                "skill": e["skill"], "task_family": e["family"], "search_arm": e["arm"], "proposer": e.get("proposer"),
                "meta_level_mechanism": {"meta_direct": "direct_replacement", "meta_rlt": "rlt_edit_budget",
                                         "random_mutation": "random_control"}[e["arm"]],
                "existing_skill_template": {"path": f"research_gym/trm_skill_gym/skills/{e['skill']}/SKILL.md",
                                            "sha256": hashlib.sha256(S.render(e["skill"]).encode()).hexdigest()},
                "incumbent_architecture": e.get("incumbent"),
                "proposed_architecture": {"raw": e.get("raw_config"), "compiled": e.get("config"), "change": e.get("change"),
                                          "dropped_by_edit_budget": e.get("dropped_by_budget"), "edit_budget": e.get("edit_budget"),
                                          "schema_error": e.get("error")},
                "rationale": e.get("rationale"),
                "falsification_criterion": e.get("falsification"),
                "predicted_result": {"dev_exact": e.get("predicted_dev_exact")},
                "actual_result": {"rung0": e.get("rung0"), "rung1": e.get("rung1"), "measured_dev_exact": e.get("measured_dev_exact"),
                                  "incumbent_dev_exact": e.get("incumbent_dev_exact"),
                                  "frozen_eval_as_H": ({k: hrow.get(k) for k in ("dev_exact", "withheld_exact", "seeds")} if hrow else None)},
                "failure_diagnosis": e.get("failure_diagnosis"),
                "revised_architecture": revised,
                "final_promotion_decision": final,
                "status": e.get("status"),
            }
            f.write(json.dumps(rec, sort_keys=True, default=float) + "\n")
            n += 1
    return n


def write_report(out: Path, main: Dict, dec: Dict, srch: Dict, txt: Dict, transfer: List[Dict], ablations: List[Dict],
                 frozen: Dict, state: Dict, packages: Dict, n_records: int, ledger: Dict, receipts: Dict, pareto: List[Dict]) -> None:
    L = []
    L.append("# TRM-SKILL-GYM-01 results\n")
    L.append("Generated by `python -m research_gym.trm_skill_gym.report` from the run directory. Preregistration: "
             "[PREREGISTRATION.md](PREREGISTRATION.md). Implementation labels: `native_trm_update` = TRM recursion "
             "(n latent steps, answer step, T cycles, deep supervision) over a small set/relation mixer backbone; "
             "`recurrent_proxy_gru` = GRU cell over the same mixer; `feedforward` = mixer stack once per visit; "
             "`heuristic_existing_skill` = the SKILL.md procedure as host code.\n")
    g = dec.get("_global", {})
    L.append("## Decision summary\n")
    L.append(f"- Skills promoted (module beats max(A,B,C) by >= {MARGIN} on frozen dev, no withheld regression vs A): "
             f"{g.get('skills_promoted') or 'none'}")
    L.append(f"- Skills where D or F beats both B and C by >= {MARGIN}: {g.get('skills_trm_beats_proxies') or 'none'}")
    L.append(f"- H (Inky-engineered) beats D in {g.get('skills_where_H_beats_D')} of 6 skills (rule: >= 4).")
    L.append(f"- Search promotions to incumbent: " + ", ".join(f"{a} {s['promoted_to_incumbent']}/{s['trained']} trained"
                                                             for a, s in srch.items()))
    L.append(f"- Skill packages exported: {list(packages) or 'none'}; GoblinBench records: {n_records}.\n")
    L.append("## Frozen evaluation: exact success (mean over seeds; fresh roots)\n")
    L.append("| skill | " + " | ".join(a.split("_")[0] for a in ARMS) + " |")
    L.append("|---|" + "---|" * len(ARMS))
    for skill in T.SKILLS:
        rows = main.get(skill, {})
        L.append(f"| {skill} dev | " + " | ".join(f3(rows.get(a, {}).get("dev_exact")) for a in ARMS) + " |")
        L.append(f"| {skill} withheld | " + " | ".join(f3(rows.get(a, {}).get("withheld_exact")) for a in ARMS) + " |")
    L.append("\nA existing skill, B feedforward, C GRU proxy, D TRM direct, E TRM matched damping, F TRM RLT, "
             "G TRM+LDT, H Inky-engineered.\n")
    L.append("## Work, calibration and resources (frozen dev, mean over seeds)\n")
    L.append("| skill | arm | impl | params | verifier calls | FLOPs/inst | ECE | adaptive exact | adaptive visits | train s | peak RSS MiB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for skill in T.SKILLS:
        for a in ARMS:
            r = main.get(skill, {}).get(a)
            if not r:
                continue
            L.append(f"| {skill} | {a} | {r['implementation']} | {r['params']} | {f3(r['verifier_calls'])} | "
                     f"{(r['flops_per_instance'] or 0) / 1e6:.2f}M | {f3(r['dev_ece'])} | {f3(r['dev_adaptive_exact'])} | "
                     f"{f3(r['dev_adaptive_visits'])} | {f3(r['train_seconds'])} | {f3(r['peak_rss_delta_mib'])} |")
    L.append("\n## Repair success (verifier acceptance; differs from exact only for repair, where exact = minimum flip set)\n")
    L.append("| arm | dev exact | dev repair success | withheld exact | withheld repair success |")
    L.append("|---|---|---|---|---|")
    for a in ARMS:
        r = main.get("repair", {}).get(a)
        if r:
            L.append(f"| {a} | {f3(r['dev_exact'])} | {f3(r['dev_repair_success'])} | {f3(r['withheld_exact'])} | {f3(r['withheld_repair_success'])} |")
    L.append("\n## Recurrence extrapolation (exact at 1 visit / trained R / 2R visits)\n")
    L.append("| skill | arm | dev 1 | dev R | dev 2R | withheld 1 | withheld R | withheld 2R |")
    L.append("|---|---|---|---|---|---|---|---|")
    for skill in T.SKILLS:
        for a in ARMS[1:]:
            r = main.get(skill, {}).get(a)
            if r:
                L.append(f"| {skill} | {a} | {f3(r['dev_exact_1visit'])} | {f3(r['dev_exact'])} | {f3(r['dev_exact_2R'])} | "
                         f"{f3(r['withheld_exact_1visit'])} | {f3(r['withheld_exact'])} | {f3(r['withheld_exact_2R'])} |")
    L.append("\n## Ablations (seed 0, frozen dev exact)\n")
    L.append("| skill | D | D shuffled supervision | D latent reset | F | F latent reset |")
    L.append("|---|---|---|---|---|---|")
    for skill in T.SKILLS:
        ab = {j["arm"]: j for j in ablations if j["skill"] == skill and j.get("status") == "ok"}
        r = main.get(skill, {})
        L.append(f"| {skill} | {f3(r.get('D_trm_direct', {}).get('dev_exact'))} | {f3(ab.get('D_shuffled_supervision', {}).get('frozen_dev', {}).get('exact'))} | "
                 f"{f3(ab.get('D_latent_reset', {}).get('frozen_dev', {}).get('exact'))} | {f3(r.get('F_trm_rlt', {}).get('dev_exact'))} | "
                 f"{f3(ab.get('F_latent_reset', {}).get('frozen_dev', {}).get('exact'))} |")
    L.append("\n## Text-only instructions vs trained modules (same first instances of frozen dev)\n")
    L.append("Text-only = Inky Mini executing the SKILL.md with no module and no verifier call; compare with visit-1 "
             "(pre-verifier) exact of the existing skill A and of modules, and with their final exact.\n")
    L.append("| skill | n | text-only | A visit1 | A final | D visit1 | D final | best module final |")
    L.append("|---|---|---|---|---|---|---|---|")
    for skill, r in txt.items():
        best = max([r.get(f"{a}_final_exact") or 0 for a in MODULE_ARMS] or [0])
        L.append(f"| {skill} | {r['n']} | {f3(r['text_only_exact'])} | {f3(r['A_visit1_exact'])} | {f3(r['A_final_exact'])} | "
                 f"{f3(r.get('D_trm_direct_visit1_exact'))} | {f3(r.get('D_trm_direct_final_exact'))} | {f3(best)} |")
    L.append("\n## Transfer (D architecture, seed 0, short target budget)\n")
    L.append("| source -> target | scratch | shared initialization | frozen latent module |")
    L.append("|---|---|---|---|")
    pairs = sorted({(j["source"], j["skill"]) for j in transfer if j.get("status") == "ok"})
    for src, tgt in pairs:
        g2 = {j["mode"]: j["frozen_dev"]["exact"] for j in transfer if j.get("source") == src and j["skill"] == tgt and j.get("status") == "ok"}
        L.append(f"| {src} -> {tgt} | {f3(g2.get('scratch'))} | {f3(g2.get('shared_initialization'))} | {f3(g2.get('frozen_latent_module'))} |")
    L.append("\n## Architecture search (external meta-repair; separate from in-module RLT answer feedback)\n")
    L.append("| arm | rounds | proposals | schema rejected | no change | trained | promoted | Inky prediction MAE | bias |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a, s in srch.items():
        L.append(f"| {a} | {s['rounds']} | {s['proposals']} | {s['schema_rejected']} | {s['no_change']} | {s['trained']} | "
                 f"{s['promoted_to_incumbent']} | {f3(s['prediction_mae'])} | {f3(s['prediction_bias'])} |")
    L.append("\nFrozen H architectures (selected on held-out development roots before the frozen evaluation):\n")
    for skill, h in frozen.get("architectures", {}).items():
        ch = dsl.diff(dsl.validate({"core": "trm"}), h["config"])
        L.append(f"- {skill}: source `{h['source']}`, dev {f3(h.get('dev_exact'))}, change vs D: `{json.dumps(ch, sort_keys=True)}`")
    fd = state.get("final_diagnosis", {})
    for arm, obj in fd.items():
        if isinstance(obj, dict) and obj.get("diagnosis"):
            L.append(f"\nInky final diagnosis ({arm}): {obj['diagnosis'][:1200]}")
    front = [p for p in pareto if p.get("pareto")]
    L.append(f"\nPareto frontier over (dev exact, params, verifier calls, visits, FLOPs): {len(front)} of {len(pareto)} archived points; "
             "see `run/search/pareto.json`.\n")
    L.append("## Per-skill decisions\n")
    for skill in T.SKILLS:
        L.append(f"- {skill}: `{json.dumps(dec.get(skill, {}), sort_keys=True, default=lambda x: round(x, 4))}`")
    L.append("\n## Resources\n")
    L.append(f"- Inky (free Inkling-2-Mini endpoint) ledger: `{json.dumps(ledger, sort_keys=True)}`")
    L.append(f"- Campaign profile: `{json.dumps(receipts.get('profile', {}), sort_keys=True)}`")
    L.append("- CPU only, one training worker, torch threads 1, 3 GiB incremental RSS cap per training run.\n")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run, out = a.run, a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    main_jobs = load_jobs(run, "main")
    ablations = load_jobs(run, "ablation")
    transfer = load_jobs(run, "transfer")
    state = rj(run / "search" / "state.json") if (run / "search" / "state.json").exists() else {}
    frozen = rj(run / "frozen_architectures.json") if (run / "frozen_architectures.json").exists() else {}
    receipts = rj(run / "receipts.json") if (run / "receipts.json").exists() else {}
    pareto = rj(run / "search" / "pareto.json") if (run / "search" / "pareto.json").exists() else []
    ledger = {}
    for p in (run / "inky_ledger.json", run / "text_only" / "ledger.json"):
        if p.exists():
            for k, v in rj(p).items():
                ledger[k] = ledger.get(k, 0) + v
    S.write_all(Path(__file__).resolve().parent / "skills")
    main_agg = aggregate_main(main_jobs)
    dec = decisions(main_agg, ablations)
    srch = search_summary(state)
    txt = paired_textonly(run, main_jobs)
    packages = export_packages(run, out, main_agg, dec, frozen, receipts)
    n_rec = goblinbench_records(run, out, state, frozen, dec, main_agg)
    summary = {"main": main_agg, "decisions": dec, "search": srch, "text_only": txt, "packages": packages,
               "goblinbench_records": n_rec, "frozen_architectures_sha256": frozen.get("sha256"),
               "receipts": receipts, "inky_ledger": ledger,
               "transfer": [{k: j.get(k) for k in ("skill", "source", "mode", "status")} | {"dev_exact": j.get("frozen_dev", {}).get("exact")}
                            for j in transfer]}
    wj(out / "summary.json", summary)
    write_report(out, main_agg, dec, srch, txt, transfer, ablations, frozen, state, packages, n_rec, ledger, receipts, pareto)
    print(json.dumps(dec.get("_global"), indent=1))


if __name__ == "__main__":
    main()
