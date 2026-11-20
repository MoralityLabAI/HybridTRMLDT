"""MCP-03 pipeline over SWE-bench repositories.

    python -m MCP.mcp03.run dataset  --config CFG --split train|dev --out DATA.json.gz
    python -m MCP.mcp03.run evaluate --config CFG [--checkpoints DIR] --out RUN [--device cpu]
    python -m MCP.mcp03.run summarize --run RUN
    python -m MCP.mcp03.run replay --run RUN [--every N]

Splits are grouped by repository: ``train`` and ``dev`` instances come only from
the config's training repositories; ``test`` instances come only from
held-out repositories. Snapshots are processed repository by repository in
creation order so the blob parse cache is reused and then released.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from MCP.mcp02 import features as FE
from MCP.mcp02.textutil import canonical_json, digest
from MCP.mcp02.run import host_receipt, peak_rss_bytes, source_receipt
from MCP.mcp02.economics import wilson
from . import GYM_VERSION
from .agent import (CODE_CONTROLS, CodeOraclePolicy, bm25_file_baseline, make_code_control, oracle_context_tokens,
                    row_item, run_code_episode)
from .env import CodeEnv
from .repo import SnapshotStore, gold_symbols, load_instances
from .tokens import describe as tokenizer_name

ROOT = Path(__file__).resolve().parents[2]
EXCLUSION_SLOT = {"none": 0, "test_path": 1, "doc_path": 2}


def select_instances(cfg: dict, split: str) -> list[dict]:
    work = Path(cfg["work_dir"])
    spec = cfg["splits"][split]
    rows = load_instances(work / spec["source"])
    rows = [r for r in rows if r["repo"] in spec["repos"]]
    exclude = set()
    for other in spec.get("exclude_ids_from", []):
        exclude |= {r["instance_id"] for r in load_instances(work / other)}
    rows = [r for r in rows if r["instance_id"] not in exclude]
    rows.sort(key=lambda r: (r["repo"], r["created_at"], r["instance_id"]))
    if spec.get("per_repo"):
        rng = random.Random(f"mcp03/{split}/{cfg['seed']}")
        picked = []
        for repo, n in spec["per_repo"].items():
            pool = [r for r in rows if r["repo"] == repo]
            picked += sorted(rng.sample(pool, min(n, len(pool))), key=lambda r: (r["created_at"], r["instance_id"]))
        rows = picked
    held = set(cfg["splits"]["test"]["repos"])
    if split != "test" and held & {r["repo"] for r in rows}:
        raise ValueError("training split overlaps held-out repositories")
    return rows


def iter_envs(cfg: dict, rows: list[dict], arms: list[str]):
    """Yield (instance, gold, envs_by_arm), one repository at a time."""
    work = Path(cfg["work_dir"])
    current_repo, store = None, None
    for inst in rows:
        if inst["repo"] != current_repo:
            store = SnapshotStore(work / "repos", work / "cache")
            current_repo = inst["repo"]
        snap = store.snapshot(inst["repo"], inst["base_commit"])
        gold = gold_symbols(snap, inst["patch"])
        if not gold["files"]:
            yield inst, gold, None
            continue
        base = CodeEnv(snap, {}, gold, arms[0])
        index = base.index
        file_index = base.file_index
        yield inst, gold, {arm: (lambda arm=arm: CodeEnv(snap, {}, gold, arm, index=index, file_index=file_index))
                           for arm in arms}


class EpsilonCodeOracle(CodeOraclePolicy):
    name, family = "epsilon_oracle_behaviour", "behaviour"

    def __init__(self, eps: float, seed: int):
        self.eps, self.seed = eps, seed

    def score(self, obs):
        rng = random.Random(f"{self.seed}/{self.episode_key}/{obs['step']}")
        base = super().score(obs)
        return [rng.random() for _ in base] if rng.random() < self.eps else [b + 1e-3 * rng.random() for b in base]


def build_dataset(cfg: dict, split: str, shard: str | None = None) -> dict:
    spec = cfg["dataset"]
    rows = select_instances(cfg, split)
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        rows = rows[k::n]
    decide, excl, skipped = [], [], 0
    rng = random.Random(f"mcp03-dataset/{split}")

    def keep(event):
        obs, labels = event["obs"], event["labels"]
        need = FE.encode_text(" ; ".join(obs["needs"]), FE.NEED_TOKENS)
        if event["kind"] == "decide":
            if not any(labels) or len(labels) < 2:
                return
            idx = list(range(len(labels)))
            pos = [i for i in idx if labels[i]]
            neg = [i for i in idx if not labels[i]]
            rng.shuffle(neg)
            k = spec["max_candidates"]
            chosen = sorted(pos[:k // 2] + neg[:k - min(len(pos), k // 2)])
            cands = [obs["candidates"][i] for i in chosen]
            decide.append({"need": need, "cands": [FE.encode_text(c["text"], FE.CAND_TOKENS) for c in cands],
                           "structs": [c["struct"] for c in cands], "labels": [labels[i] for i in chosen]})
        else:
            cands = [row_item(r, 1) for r in event["rows"]]
            excl.append({"need": need, "cands": [FE.encode_text(c["text"], FE.CAND_TOKENS) for c in cands],
                         "structs": [c["struct"] for c in cands],
                         "labels": [EXCLUSION_SLOT[l] for l in event["labels"]]})

    episodes = 0
    for inst, gold, envs in iter_envs(cfg, rows, spec["arms"]):
        if envs is None:
            skipped += 1
            continue
        for arm in spec["arms"]:
            for eps in spec["epsilons"]:
                run_code_episode(envs[arm](), inst["problem_statement"], policy=EpsilonCodeOracle(eps, cfg["seed"]),
                                 max_tool_calls=spec["max_tool_calls"], budgets=(), record=keep)
                episodes += 1
    return {"decide": decide, "rows": excl, "meta": {
        "split": split, "shard": shard, "instances": len(rows), "skipped_no_python_gold": skipped, "episodes": episodes,
        "repos": sorted({r["repo"] for r in rows}), "instance_ids": [r["instance_id"] for r in rows],
        "decide_states": len(decide), "row_states": len(excl), "encoder": FE.ENCODER_VERSION}}


class LearnedCodePolicy:
    """Wrap an MCP-02 RouterPolicy; map its exclusion slots onto MCP-03 path predicates."""

    def __init__(self, inner):
        self.inner = inner
        self.name, self.family = inner.name, inner.family

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def code_exclusions(self, obs: dict, rows: list[dict]) -> list[tuple[str, str]]:
        import torch
        model = self.inner.model
        if model.exclusion_head is None:
            from .agent import symbolic_code_exclusions
            return symbolic_code_exclusions(rows)
        cands = [row_item(r, 1) for r in rows]
        ids, segs, structs = FE.encode_observation(dict(obs, candidates=cands), model.encoder)
        with torch.inference_mode():
            _s, logits = model(*self.inner._tensors(ids, segs, structs))
        probs = logits.float().softmax(-1).cpu().tolist()
        self.inner._cost["flops"] += self.inner._flops["flops"] * len(ids)
        names = {1: "test_path", 2: "doc_path"}
        out = []
        for row, p in zip(rows, probs):
            label = max(range(len(p)), key=lambda i: p[i])
            if label in names and p[label] >= self.inner.threshold:
                out.append((row["node_id"], names[label]))
        return out


def load_policies(cfg: dict, checkpoints: Path | None, device: str, names: set | None) -> dict:
    out = {n: make_code_control(n) for n in cfg["evaluation"]["controls"] if names is None or n in names}
    if checkpoints is None:
        return out
    from MCP.mcp02.models import RouterPolicy, file_sha256, load
    for name in cfg["models"]:
        path = checkpoints / f"{name}.pt"
        if (names is not None and name not in names) or not path.is_file():
            continue
        receipt_path = path.with_suffix(".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
        receipt.pop("dataset", None)
        out[name] = LearnedCodePolicy(RouterPolicy(name, load(str(path), device), device=device,
                                                   checkpoint=file_sha256(str(path)), receipt=receipt))
    return out


def evaluate(args) -> int:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ev = cfg["evaluation"]
    rows = select_instances(cfg, "test")
    if args.limit:
        rows = rows[:args.limit]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        rows = rows[k::n]
    policies = load_policies(cfg, args.checkpoints, args.device, set(args.policies) if args.policies else None)
    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    eps_path = args.out / "episodes.jsonl"
    if eps_path.is_file():
        for line in eps_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["instance_id"], r["arm"], r["policy_name"], r["max_tool_calls"]))
    base_path = args.out / "baselines.jsonl"
    base_done = set()
    if base_path.is_file():
        base_done = {json.loads(l)["instance_id"] for l in base_path.read_text(encoding="utf-8").splitlines() if l}
    budgets = tuple(ev["budgets"])
    receipts = {}
    started = time.perf_counter()
    with open(eps_path, "a", encoding="utf-8", newline="\n") as eps, \
            open(args.out / "tapes.jsonl", "a", encoding="utf-8", newline="\n") as tapes, \
            open(base_path, "a", encoding="utf-8", newline="\n") as bases:
        for n, (inst, gold, envs) in enumerate(iter_envs(cfg, rows, ev["arms"])):
            if envs is None:
                continue
            if inst["instance_id"] not in base_done:
                env = envs[ev["arms"][0]]()
                b = bm25_file_baseline(env, inst["problem_statement"], tuple(ev["bm25_file_budgets"]))
                bases.write(canonical_json({"instance_id": inst["instance_id"], "repo": inst["repo"],
                                            "gold": gold, "bm25_files": b, "context": oracle_context_tokens(env)})
                            + "\n")
            for arm in ev["arms"]:
                for name, policy in policies.items():
                    for budget in ev["max_tool_calls"]:
                        key = (inst["instance_id"], arm, name, budget)
                        if key in done:
                            continue
                        ep = run_code_episode(envs[arm](), inst["problem_statement"], policy=policy,
                                              max_tool_calls=budget, budgets=budgets, prefilter_k=ev["prefilter_k"],
                                              compose=ev.get("composed_packets", False))
                        tape = ep.pop("tape")
                        desc = ep.pop("policy")
                        receipts[name] = desc
                        ep.update(instance_id=inst["instance_id"], policy_name=name, policy_family=desc["family"],
                                  max_tool_calls=budget, recurrent_depth=desc.get("recurrent_depth"))
                        eps.write(canonical_json(ep) + "\n")
                        tapes.write(canonical_json({"instance_id": inst["instance_id"], "arm": arm, "policy": name,
                                                    "max_tool_calls": budget, "tape": tape}) + "\n")
            eps.flush()
            tapes.flush()
            bases.flush()
            print(f"[{n + 1}/{len(rows)}] {inst['instance_id']} {time.perf_counter() - started:.0f}s", flush=True)
    path = args.out / "model_receipts.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    old.update(receipts)
    path.write_text(json.dumps(old, indent=1, default=str) + "\n", encoding="utf-8")
    host = host_receipt(args.device)
    host["peak_rss_bytes"] = peak_rss_bytes()
    host["tokenizer"] = tokenizer_name()
    (args.out / "host.json").write_text(json.dumps(host, indent=1) + "\n", encoding="utf-8")
    (args.out / "source_manifest.json").write_text(json.dumps(source_receipt(), indent=1) + "\n", encoding="utf-8")
    (args.out / "run_config.json").write_text(json.dumps({"gym_version": GYM_VERSION, "config": cfg,
                                                          "device": args.device}, indent=1) + "\n", encoding="utf-8")
    return 0


def recompute_baselines(args) -> int:
    """Rewrite baselines.jsonl (BM25 whole files and BM25 path maps, gold context sizes) for the test split."""
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ev = cfg["evaluation"]
    rows = select_instances(cfg, "test")
    tmp = args.out / "baselines.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8", newline=chr(10)) as out:
        for inst, gold, envs in iter_envs(cfg, rows, ev["arms"][:1]):
            if envs is None:
                continue
            env = envs[ev["arms"][0]]()
            b = bm25_file_baseline(env, inst["problem_statement"], tuple(ev["bm25_file_budgets"]))
            out.write(canonical_json({"instance_id": inst["instance_id"], "repo": inst["repo"], "gold": gold,
                                      "bm25_files": b, "context": oracle_context_tokens(env)}) + chr(10))
    os.replace(tmp, args.out / "baselines.jsonl")
    return 0


def merge_runs(inputs: list[Path], out: Path) -> int:
    """Concatenate sharded evaluation runs (episodes, tapes, baselines) and merge their receipts."""
    import shutil
    out.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "tapes.jsonl", "baselines.jsonl"):
        with open(out / name, "w", encoding="utf-8", newline=chr(10)) as dst:
            for run in inputs:
                if (run / name).is_file():
                    dst.write((run / name).read_text(encoding="utf-8"))
    receipts = {}
    for run in inputs:
        if (run / "model_receipts.json").is_file():
            receipts.update(json.loads((run / "model_receipts.json").read_text(encoding="utf-8")))
    (out / "model_receipts.json").write_text(json.dumps(receipts, indent=1, default=str) + chr(10), encoding="utf-8")
    hosts = [json.loads((r / "host.json").read_text(encoding="utf-8")) for r in inputs if (r / "host.json").is_file()]
    if hosts:
        host = dict(hosts[0], shards=len(inputs), peak_rss_bytes_by_shard=[h.get("peak_rss_bytes") for h in hosts])
        (out / "host.json").write_text(json.dumps(host, indent=1) + chr(10), encoding="utf-8")
    for name in ("source_manifest.json", "run_config.json"):
        if (inputs[0] / name).is_file():
            shutil.copyfile(inputs[0] / name, out / name)
    return 0


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(run: Path) -> dict:
    def read(name):
        p = run / name
        if not p.is_file():
            p = run / (name + ".gz")
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as stream:
            return [json.loads(l) for l in stream if l.strip()]
    eps = read("episodes.jsonl")
    bases = {b["instance_id"]: b for b in read("baselines.jsonl")}
    groups: dict[tuple, list] = {}
    for e in eps:
        groups.setdefault((e["arm"], e["policy_name"], e["max_tool_calls"]), []).append(e)
    rows = []
    for (arm, name, budget), group in sorted(groups.items()):
        row = {"arm": arm, "policy": name, "family": group[0]["policy_family"], "max_tool_calls": budget,
               "recurrent_depth": group[0].get("recurrent_depth"), "instances": len(group),
               "tool_calls_mean": _mean([e["tool_calls"] for e in group]),
               "retrieval_result_tokens_mean": _mean([e["compute"]["result_tokens"] for e in group]),
               "router_tokens_mean": _mean([e["compute"]["router_tokens"] for e in group]),
               "router_flops_mean": _mean([e["compute"]["flops"] for e in group]),
               "router_ms_per_decision": 1000 * sum(e["compute"]["router_seconds"] for e in group) /
               max(1, sum(e["compute"]["decisions"] for e in group)),
               "certified_exclusions_containing_gold": sum(e["private_checks"]["certified_exclusions_containing_gold"]
                                                           for e in group),
               "ldt_excluded": sum(e["environment_stats"]["ldt_excluded"] for e in group), "budgets": {}}
        for b in group[0]["packets"]:
            pk = [e["packets"][b] for e in group]
            allf = sum(p["grade"]["all_files"] for p in pk)
            row["budgets"][b] = {"packet_tokens_mean": _mean([p["tokens"] for p in pk]),
                                 "file_recall_mean": _mean([p["grade"]["file_recall"] for p in pk]),
                                 "all_files_hit": allf, "all_files_ci95": wilson(allf, len(pk)),
                                 "symbol_recall_mean": _mean([p["grade"]["symbol_recall"] for p in pk]),
                                 "all_symbols_hit": sum(p["grade"]["all_symbols"] for p in pk)}
        repos: dict[str, list] = {}
        for e in group:
            repos.setdefault(e["repo"], []).append(e["packets"]["2048"]["grade"]["file_recall"]
                                                   if "2048" in e["packets"] else None)
        row["file_recall_2048_by_repo"] = {r: _mean([x for x in v if x is not None]) for r, v in sorted(repos.items())}
        rows.append(row)
    baseline = {}
    for b in next(iter(bases.values()))["bm25_files"]["packets"] if bases else []:
        pk = [x["bm25_files"]["packets"][b] for x in bases.values()]
        baseline[b] = {"tokens_mean": _mean([p["tokens"] for p in pk]),
                       "file_recall_mean": _mean([p["grade"]["file_recall"] for p in pk]),
                       "all_files_hit": sum(p["grade"]["all_files"] for p in pk), "instances": len(pk)}
    maps = {}
    first = next(iter(bases.values()), {})
    for k in first.get("bm25_files", {}).get("path_maps", {}):
        pk = [x["bm25_files"]["path_maps"][k] for x in bases.values() if "path_maps" in x["bm25_files"]]
        maps[k] = {"tokens_mean": _mean([p["tokens"] for p in pk]),
                   "file_recall_mean": _mean([p["grade"]["file_recall"] for p in pk]),
                   "all_files_hit": sum(p["grade"]["all_files"] for p in pk), "instances": len(pk)}
    context = {"gold_files_tokens_mean": _mean([x["context"]["gold_files_tokens"] for x in bases.values()]),
               "repo_python_bytes_mean": _mean([x["context"]["repo_python_bytes"] for x in bases.values()]),
               "repo_python_files_mean": _mean([x["context"]["repo_python_files"] for x in bases.values()])}
    summary = {"gym_version": GYM_VERSION, "episodes": len(eps), "instances": len(bases), "groups": rows,
               "bm25_file_baseline": baseline, "bm25_path_map_baseline": maps, "context": context,
               "episodes_sha256": hashlib.sha256(canonical_json(eps).encode()).hexdigest()}
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def replay(run: Path, every: int = 1) -> dict:
    meta = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    ev = cfg["evaluation"]
    def read(name):
        p = run / name if (run / name).is_file() else run / (name + ".gz")
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as stream:
            return [json.loads(l) for l in stream if l.strip()]
    tapes = {(t["instance_id"], t["arm"], t["policy"], t["max_tool_calls"]): t["tape"] for t in read("tapes.jsonl")}
    eps = read("episodes.jsonl")
    sampled = [e for i, e in enumerate(eps) if i % every == 0]
    wanted = {e["instance_id"] for e in sampled}
    rows = [r for r in select_instances(cfg, "test") if r["instance_id"] in wanted]
    by_id = {}
    for e in sampled:
        by_id.setdefault(e["instance_id"], []).append(e)
    checked = mismatched = 0
    host = json.loads((run / "host.json").read_text(encoding="utf-8")) if (run / "host.json").is_file() else {}
    same_tokenizer = host.get("tokenizer") == tokenizer_name()
    for inst, gold, envs in iter_envs(cfg, rows, ev["arms"]):
        for e in by_id.get(inst["instance_id"], []):
            tape = tapes[(e["instance_id"], e["arm"], e["policy_name"], e["max_tool_calls"])]
            budgets = tuple(sorted({int(b.lstrip("c")) for b in e["packets"]}))
            out = run_code_episode(envs[e["arm"]](), inst["problem_statement"], policy=make_code_control("sham_random"),
                                   max_tool_calls=e["max_tool_calls"], budgets=budgets, prefilter_k=ev["prefilter_k"],
                                   tape=tape, compose=any(b.startswith("c") for b in e["packets"]))
            same = (out["results_sha256"] == e["results_sha256"] and out["reads"] == e["reads"]
                    and out["tool_calls"] == e["tool_calls"])
            if same_tokenizer:
                same = same and out["packets"] == e["packets"]
            checked += 1
            mismatched += not same
    return {"checked": checked, "mismatched": mismatched, "packets_compared": same_tokenizer}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--config", type=Path, required=True)
    d.add_argument("--split", choices=("train", "dev"), required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--shard", help="k/n: build every n-th instance starting at k")
    mg = sub.add_parser("merge")
    mg.add_argument("--inputs", type=Path, nargs="+", required=True)
    mg.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--checkpoints", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--device", default="cpu")
    e.add_argument("--policies", nargs="*")
    e.add_argument("--limit", type=int)
    e.add_argument("--shard", help="k/n: evaluate every n-th test instance starting at k")
    mr = sub.add_parser("merge-runs")
    mr.add_argument("--inputs", type=Path, nargs="+", required=True)
    mr.add_argument("--out", type=Path, required=True)
    bl = sub.add_parser("baselines")
    bl.add_argument("--config", type=Path, required=True)
    bl.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--run", type=Path, required=True)
    r = sub.add_parser("replay")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--every", type=int, default=1)
    args = p.parse_args(argv)
    if args.cmd == "dataset":
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        data = build_dataset(cfg, args.split, args.shard)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.out, "wt", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
        print(json.dumps({k: v for k, v in data["meta"].items() if k != "instance_ids"}, indent=1))
        return 0
    if args.cmd == "merge":
        parts = [json.load(gzip.open(p, "rt", encoding="utf-8")) for p in args.inputs]
        meta = dict(parts[0]["meta"], shard=None, merged_from=[str(p) for p in args.inputs])
        for key in ("instances", "skipped_no_python_gold", "episodes", "decide_states", "row_states"):
            meta[key] = sum(p["meta"][key] for p in parts)
        meta["instance_ids"] = sorted(i for p in parts for i in p["meta"]["instance_ids"])
        meta["repos"] = sorted({r for p in parts for r in p["meta"]["repos"]})
        data = {"decide": [s for p in parts for s in p["decide"]], "rows": [s for p in parts for s in p["rows"]],
                "meta": meta}
        with gzip.open(args.out, "wt", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
        print(json.dumps({k: v for k, v in meta.items() if k != "instance_ids"}))
        return 0
    if args.cmd == "evaluate":
        return evaluate(args)
    if args.cmd == "baselines":
        return recompute_baselines(args)
    if args.cmd == "merge-runs":
        return merge_runs(args.inputs, args.out)
    if args.cmd == "summarize":
        s = summarize(args.run)
        for row in s["groups"]:
            b = row["budgets"].get("2048", {})
            print(f"{row['arm']:11s} {row['policy']:28s} calls {row['tool_calls_mean']:.1f} "
                  f"pkt {b.get('packet_tokens_mean', 0):.0f} fileR {b.get('file_recall_mean', 0):.3f} "
                  f"allF {b.get('all_files_hit')}/{row['instances']} symR {b.get('symbol_recall_mean', 0):.3f}")
        print("bm25 files", s["bm25_file_baseline"])
        return 0
    result = replay(args.run, args.every)
    print(json.dumps(result))
    return 0 if result["mismatched"] == 0 and result["checked"] else 1


if __name__ == "__main__":
    sys.exit(main())
