"""MCP-02 experiment runner.

Commands::

    python -m MCP.mcp02.run evaluate --config CFG [--checkpoints DIR] --out RUN [--controls-only]
    python -m MCP.mcp02.run summarize --run RUN
    python -m MCP.mcp02.run replay --run RUN [--every N]
    python -m MCP.mcp02.run memprobe --checkpoint PT

``evaluate`` runs the main grid and the registered sweeps, appending one lean
episode record per line and a separate decision tape. Controls need only the
standard library; learned routers need torch and their checkpoints. ``replay``
re-executes episodes from their tapes without torch and checks every recorded
outcome, so CI can verify a GPU or CPU run's environment trajectories.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import GYM_VERSION
from .agent import CONTROL_POLICIES, Policy, make_control, run_episode
from .economics import paired_difference, summarize_group
from .textutil import canonical_json, digest
from .world import make_cases

ROOT = Path(__file__).resolve().parents[2]


def _open(path: Path, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else open(path, mode, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with _open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_receipt() -> dict:
    paths = sorted((ROOT / "MCP" / "mcp02").rglob("*.py")) + sorted((ROOT / "MCP" / "mcp02" / "configs").glob("*.json"))
    paths += [ROOT / p for p in ("research_gym/core/hybrid.py", "research_gym/core/typed_soundness.py",
                                 "MCP/transport.py", "MCP/serve.py", "MCP/local_model.py")]
    # LF-normalised so Windows (autocrlf) and Linux checkouts of one commit hash alike.
    files = {str(p.relative_to(ROOT)).replace("\\", "/"):
             hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths if p.is_file()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "MCP"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "mcp_tree_dirty": dirty, "files": files, "hash": "sha256 of LF-normalised bytes",
            "source_digest": digest(files)}


def host_receipt(device: str) -> dict:
    info: dict[str, Any] = {"platform": platform.platform(), "python": sys.version.split()[0],
                            "processor": platform.processor(), "logical_cpus": os.cpu_count(), "device": device}
    try:
        import torch
        info.update(torch=torch.__version__, torch_threads=torch.get_num_threads())
    except ImportError:
        info["torch"] = None
    try:
        info["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,enforced.power.limit",
             "--format=csv,noheader"], text=True, timeout=20).strip()
    except (OSError, subprocess.SubprocessError):
        info["gpu"] = None
    return info


def peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except ImportError:
        return None


def memprobe(checkpoint: str, device: str = "cpu") -> dict:
    """Peak process memory around one 32-candidate scoring pass, in a fresh process."""
    import torch
    from . import features as FE
    from .models import load, parameter_counts
    model = load(checkpoint, device)
    before = peak_rss_bytes()
    ids = torch.full((32, FE.SEQ_LEN), FE.OFFSET + 3, dtype=torch.long, device=device)
    segs = torch.zeros_like(ids)
    segs[:, FE.NEED_TOKENS + 2:] = 1
    structs = torch.zeros(32, FE.STRUCT_DIM, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(ids, segs, structs)
    after = peak_rss_bytes()
    out = {"checkpoint": checkpoint, "device": device, "peak_rss_before_forward": before,
           "peak_rss_after_forward": after,
           "forward_peak_rss_delta": (after - before) if before is not None and after is not None else None,
           "weights_bytes_fp32": parameter_counts(model)["bytes_fp32"], "candidates": 32,
           "sequence_length": FE.SEQ_LEN}
    if device == "cuda":
        out["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
    return out


# -- policies --------------------------------------------------------------------------
def load_policies(cfg: dict, checkpoints: Path | None, device: str, names: set[str] | None) -> dict[str, Policy]:
    policies: dict[str, Policy] = {}
    for name in cfg["evaluation"]["controls"]:
        if names is None or name in names:
            policies[name] = make_control(name)
    if checkpoints is None:
        return policies
    from .models import RouterPolicy, file_sha256, load
    for name in cfg["models"]:
        path = checkpoints / f"{name}.pt"
        if (names is not None and name not in names) or not path.is_file():
            continue
        receipt_path = path.with_suffix(".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
        receipt.pop("dataset", None)
        policies[name] = RouterPolicy(name, load(str(path), device), device=device,
                                      checkpoint=file_sha256(str(path)), receipt=receipt)
    return policies


def jobs(cfg: dict, available: list[str]) -> list[dict]:
    ev = cfg["evaluation"]
    out = []
    for size in ev["catalog_sizes"]:
        for arm, layout in ev["arm_configs"]:
            for budget in ev["max_tool_calls"]:
                for name in available:
                    out.append({"experiment": "main", "catalog_size": size, "arm": arm, "layout": layout,
                                "policy": name, "max_tool_calls": budget, "depth": None, "worlds": ev["worlds"]})
    sweep = ev.get("depth_sweep")
    if sweep:
        for name in sweep["models"]:
            if name not in available:
                continue
            for depth in sweep["depths"]:
                for arm, layout in sweep["arm_configs"]:
                    out.append({"experiment": "depth", "catalog_size": sweep["catalog_size"], "arm": arm,
                                "layout": layout, "policy": name, "max_tool_calls": ev["max_tool_calls"][0],
                                "depth": depth, "worlds": ev["worlds"]})
    sweep = ev.get("deliberation_sweep")
    if sweep:
        for name in sweep["policies"]:
            if name not in available:
                continue
            for budget in sweep["max_tool_calls"]:
                for arm, layout in sweep["arm_configs"]:
                    out.append({"experiment": "deliberation", "catalog_size": sweep["catalog_size"], "arm": arm,
                                "layout": layout, "policy": name, "max_tool_calls": budget, "depth": None,
                                "worlds": ev["worlds"]})
    sweep = ev.get("layout_sweep")
    if sweep:
        for name in sweep["policies"]:
            if name not in available:
                continue
            for arm in sweep["arms"]:
                for layout in sweep["layouts"]:
                    out.append({"experiment": "layout", "catalog_size": sweep["catalog_size"], "arm": arm,
                                "layout": layout, "policy": name, "max_tool_calls": ev["max_tool_calls"][0],
                                "depth": None, "worlds": sweep["worlds"]})
    return out


def job_key(job: dict) -> str:
    return canonical_json({k: job[k] for k in ("experiment", "catalog_size", "arm", "layout", "policy",
                                               "max_tool_calls", "depth", "worlds")})


def evaluate(args) -> int:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ev = cfg["evaluation"]
    names = set(args.policies) if args.policies else None
    policies = load_policies(cfg, None if args.controls_only else args.checkpoints, args.device, names)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes_path, tapes_path = args.out / "episodes.jsonl", args.out / "tapes.jsonl"
    done: set[tuple[str, str]] = set()
    if episodes_path.is_file():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["job_key"], rec["case_id"]))
    case_cache: dict[tuple[int, int], list[dict]] = {}
    plan = jobs(cfg, list(policies))
    receipts = {}
    started = time.perf_counter()
    with open(episodes_path, "a", encoding="utf-8", newline="\n") as eps, \
            open(tapes_path, "a", encoding="utf-8", newline="\n") as tapes:
        for j, job in enumerate(plan):
            key = job_key(job)
            policy = policies[job["policy"]]
            if hasattr(policy, "set_depth"):
                policy.set_depth(job["depth"])
            receipts[job["policy"]] = policy.describe()
            cache_key = (job["catalog_size"], job["worlds"])
            if cache_key not in case_cache:
                case_cache[cache_key] = make_cases(ev["split"], ev["seed"], job["worlds"], job["catalog_size"])
            for index, case in enumerate(case_cache[cache_key]):
                if (key, case["id"]) in done:
                    continue
                ep = run_episode(case, arm=job["arm"], layout=job["layout"], policy=policy,
                                 max_tool_calls=job["max_tool_calls"], prefilter_k=ev["prefilter_k"],
                                 handoff_budget_bytes=ev["handoff_budget_bytes"], transport=args.transport,
                                 stdio_options={"seed": ev["seed"], "index": index})
                tape = ep.pop("decisions")
                ep.pop("handoff")
                desc = ep.pop("policy")
                ep.update(job_key=key, job=job, case_index=index, policy_name=job["policy"],
                          policy_family=desc["family"], depth=job["depth"],
                          recurrent_depth=desc.get("recurrent_depth"))
                eps.write(canonical_json(ep) + "\n")
                tapes.write(canonical_json({"job_key": key, "case_id": case["id"], "tape": tape}) + "\n")
            eps.flush()
            tapes.flush()
            if hasattr(policy, "set_depth"):
                policy.set_depth(None)
            print(f"[{j + 1}/{len(plan)}] {job['experiment']} {job['policy']} {job['arm']}/{job['layout']} "
                  f"size={job['catalog_size']} budget={job['max_tool_calls']} depth={job['depth']} "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    receipt_path = args.out / "model_receipts.json"
    old = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    old.update(receipts)
    receipt_path.write_text(json.dumps(old, indent=1, default=str) + "\n", encoding="utf-8")
    (args.out / "host.json").write_text(json.dumps(host_receipt(args.device), indent=1) + "\n", encoding="utf-8")
    (args.out / "source_manifest.json").write_text(json.dumps(source_receipt(), indent=1) + "\n", encoding="utf-8")
    meta = {"gym_version": GYM_VERSION, "config": cfg, "config_path": str(args.config).replace("\\", "/"),
            "device": args.device, "transport": args.transport}
    (args.out / "run_config.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return 0


# -- summary -----------------------------------------------------------------------------
def _prices(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(run: Path, prices_path: Path) -> dict:
    prices_all = _prices(prices_path)
    base = {k: v for k, v in prices_all.items() if not isinstance(v, (dict, str))}
    meta = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    device = meta["device"]
    path = run / "episodes.jsonl" if (run / "episodes.jsonl").is_file() else run / "episodes.jsonl.gz"
    episodes = read_jsonl(path)
    groups: dict[str, list[dict]] = {}
    for e in episodes:
        groups.setdefault(e["job_key"], []).append(e)
    rows = []
    for key, group in sorted(groups.items()):
        job = group[0]["job"]
        row = {**job, "policy_family": group[0]["policy_family"], "recurrent_depth": group[0]["recurrent_depth"],
               **summarize_group(group, base, device)}
        row["cost_per_success_by_scenario"] = {}
        for name, override in prices_all.get("scenarios", {}).items():
            prices = dict(base, **override)
            row["cost_per_success_by_scenario"][name] = summarize_group(group, prices, device)["cost_per_success_usd"]
        rows.append(row)
    # Paired comparisons on the main grid: every policy against the two simplest controls.
    paired = []
    main = [e for e in episodes if e["job"]["experiment"] == "main"]
    cells: dict[tuple, dict[str, dict[str, dict]]] = {}
    for e in main:
        cell = (e["job"]["catalog_size"], e["job"]["arm"], e["job"]["layout"], e["job"]["max_tool_calls"])
        cells.setdefault(cell, {}).setdefault(e["policy_name"], {})[e["case_id"]] = e
    from .economics import episode_cost
    for cell, by_policy in sorted(cells.items()):
        for reference in ("deterministic_rules", "lexical_bm25"):
            if reference not in by_policy:
                continue
            ref = by_policy[reference]
            for name, eps in sorted(by_policy.items()):
                if name == reference:
                    continue
                ids = sorted(set(ref) & set(eps))
                paired.append({
                    "catalog_size": cell[0], "arm": cell[1], "layout": cell[2], "max_tool_calls": cell[3],
                    "policy": name, "reference": reference,
                    "success_difference": paired_difference([float(eps[i]["accepted"]) for i in ids],
                                                            [float(ref[i]["accepted"]) for i in ids], seed=1),
                    "tool_call_difference": paired_difference([float(eps[i]["tool_calls"]) for i in ids],
                                                              [float(ref[i]["tool_calls"]) for i in ids], seed=2),
                    "cost_difference_usd": paired_difference(
                        [episode_cost(eps[i], base, device)["total_usd"] for i in ids],
                        [episode_cost(ref[i], base, device)["total_usd"] for i in ids], seed=3)})
    by_family: dict[str, dict[str, list[int]]] = {}
    for e in main:
        by_family.setdefault(e["policy_name"], {}).setdefault(e["family"], [0, 0])
        cell = by_family[e["policy_name"]][e["family"]]
        cell[0] += int(e["accepted"])
        cell[1] += 1
    summary = {"gym_version": GYM_VERSION, "episodes": len(episodes), "groups": rows, "paired": paired,
               "success_by_family_main": by_family, "prices": prices_all, "device": device,
               "episodes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "accounting": {"failure_accounting": "every assigned episode contributes cost; cost per success "
                                                    "divides by accepted episodes",
                              "token_source": "router tokens are hashed-word tokens of the shared encoder; "
                                              "handoff tokens use the configured bytes-per-token estimator",
                              "flops_source": "torch FlopCounterMode per candidate times candidates scored",
                              "latency_source": "wall clock per router decision on the recorded host"}}
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    return summary


# -- replay ------------------------------------------------------------------------------
class TapeOnly(Policy):
    name, family = "tape_replay", "replay"

    def __init__(self, proposer: str):
        self.proposer = proposer

    def score(self, obs):
        raise RuntimeError("tape replay supplies decisions")


def replay(run: Path, every: int = 1, limit: int | None = None) -> dict:
    meta = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    ev = meta["config"]["evaluation"]
    ep_path = run / "episodes.jsonl" if (run / "episodes.jsonl").is_file() else run / "episodes.jsonl.gz"
    tp_path = run / "tapes.jsonl" if (run / "tapes.jsonl").is_file() else run / "tapes.jsonl.gz"
    tapes = {(t["job_key"], t["case_id"]): (t["tape"] if "tape" in t else {"llm_responses": t["llm_responses"]})
             for t in read_jsonl(tp_path)}
    episodes = read_jsonl(ep_path)
    cache: dict[tuple, list[dict]] = {}
    checked = mismatched = 0
    failures = []
    for i, rec in enumerate(episodes):
        if i % every:
            continue
        if limit is not None and checked >= limit:
            break
        job = rec["job"]
        key = (job["catalog_size"], job["worlds"])
        if key not in cache:
            cache[key] = make_cases(ev["split"], ev["seed"], job["worlds"], job["catalog_size"])
        case = cache[key][rec["case_index"]]
        tape = tapes[(rec["job_key"], rec["case_id"])]
        if isinstance(tape, dict) and "llm_responses" in tape:
            from .llm import ScriptedModel, run_llm_episode
            out = run_llm_episode(case, arm=job["arm"], layout=job["layout"],
                                  model=ScriptedModel(tape["llm_responses"]), max_steps=job["max_tool_calls"],
                                  handoff_budget_bytes=ev["handoff_budget_bytes"])
            fields = ("accepted", "tool_calls", "handoff_utf8_bytes", "results_sha256", "environment_stats")
            same = all(out[f] == rec[f] for f in fields)
        else:
            out = run_episode(case, arm=job["arm"], layout=job["layout"], policy=TapeOnly("tape"),
                              max_tool_calls=job["max_tool_calls"], prefilter_k=ev["prefilter_k"],
                              handoff_budget_bytes=ev["handoff_budget_bytes"], tape=tape)
            fields = ("accepted", "tool_calls", "handoff_utf8_bytes", "results_sha256", "failure",
                      "environment_stats")
            same = all(out[f] == rec[f] for f in fields) and out["compute"]["flops"] == rec["compute"]["flops"]
        checked += 1
        if not same:
            mismatched += 1
            failures.append({"job_key": rec["job_key"], "case_id": rec["case_id"],
                             "diff": {f: [rec[f], out[f]] for f in fields if out[f] != rec[f]}})
    return {"checked": checked, "mismatched": mismatched, "failures": failures[:10]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--checkpoints", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--device", default="cpu")
    e.add_argument("--transport", choices=("inprocess", "stdio"), default="inprocess")
    e.add_argument("--controls-only", action="store_true")
    e.add_argument("--policies", nargs="*")
    e.add_argument("--cuda-memory-fraction", type=float)
    s = sub.add_parser("summarize")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--prices", type=Path, default=ROOT / "MCP/mcp02/configs/economics.json")
    r = sub.add_parser("replay")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--every", type=int, default=1)
    r.add_argument("--limit", type=int)
    m = sub.add_parser("memprobe")
    m.add_argument("--checkpoint", required=True)
    m.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.cmd == "evaluate":
        if args.cuda_memory_fraction and args.device == "cuda":
            import torch
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        return evaluate(args)
    if args.cmd == "summarize":
        summary = summarize_run(args.run, args.prices)
        for row in summary["groups"]:
            if row["experiment"] == "main":
                print(f"{row['catalog_size']:5d} {row['arm']:12s} {row['layout']:5s} {row['policy']:34s} "
                      f"{row['accepted']:3d}/{row['episodes']:<3d} calls/ep {row['tool_calls_mean']:.1f} "
                      f"cost/success {row['cost_per_success_usd']}")
        return 0
    if args.cmd == "replay":
        result = replay(args.run, args.every, args.limit)
        print(json.dumps(result, indent=1))
        return 0 if result["mismatched"] == 0 and result["checked"] > 0 else 1
    print(json.dumps(memprobe(args.checkpoint, args.device)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
