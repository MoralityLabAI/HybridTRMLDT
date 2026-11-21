"""Build imitation datasets from the training split and fit learned routers.

Labels come from ``OracleView``: a candidate is positive when it leads to a
document the plan currently needs. Behaviour policies are the oracle with
epsilon-random exploration, so off-path states appear in the data. Only the
``train`` split is used for fitting; ``dev`` gives a held-out ranking check.
Test worlds never enter this module.

Usage (torch environment)::

    python -m MCP.mcp02.train dataset --split train --worlds 320 --out WORK/train.pt
    python -m MCP.mcp02.train fit --config MCP/mcp02/configs/pilot_cpu.json --arch-name trm_d4 ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from . import features as FE
from .agent import OraclePolicy, run_episode
from .world import make_cases


class EpsilonOracle(OraclePolicy):
    name, family = "epsilon_oracle_behaviour", "behaviour"

    def __init__(self, eps: float, seed: int):
        super().__init__()
        self.eps, self.seed = eps, seed

    def score(self, obs):
        rng = random.Random(f"{self.seed}/{self.episode_key}/{obs['step']}")
        base = super().score(obs)
        if rng.random() < self.eps:
            return [rng.random() for _ in base]
        return [b + 1e-3 * rng.random() for b in base]


def build_dataset(split: str, worlds: int, seed: int, catalog_size: int, arm_configs: list[list[str]],
                  epsilons: list[float], max_candidates: int = 16, max_tool_calls: int = 40) -> dict:
    """Return encoded decision states and exclusion-row states (stdlib lists)."""
    if split == "test":
        raise ValueError("test worlds must not be used for fitting")
    cases = make_cases(split, seed, worlds, catalog_size)
    rng = random.Random(f"dataset/{split}/{seed}")
    decide, rows = [], []
    episodes = 0

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
            chosen = sorted(pos[:max_candidates // 2] + neg[:max_candidates - min(len(pos), max_candidates // 2)])
            cands = [obs["candidates"][i] for i in chosen]
            decide.append({"need": need, "cands": [FE.encode_text(c["text"], FE.CAND_TOKENS) for c in cands],
                           "structs": [c["struct"] for c in cands], "labels": [labels[i] for i in chosen],
                           "family": event.get("family")})
        else:
            cands = FE.rows_as_candidates(event["rows"], 1)
            rows.append({"need": need, "cands": [FE.encode_text(c["text"], FE.CAND_TOKENS) for c in cands],
                         "structs": [c["struct"] for c in cands],
                         "labels": [FE.EXCLUSION_LABELS.index(l) for l in event["labels"]]})

    for case in cases:
        for arm, layout in arm_configs:
            for eps in epsilons:
                run_episode(case, arm=arm, layout=layout, policy=EpsilonOracle(eps, seed),
                            max_tool_calls=max_tool_calls, record=keep)
                episodes += 1
    return {"decide": decide, "rows": rows, "meta": {
        "split": split, "worlds": worlds, "seed": seed, "catalog_size": catalog_size,
        "arm_configs": arm_configs, "epsilons": epsilons, "episodes": episodes,
        "max_candidates": max_candidates, "encoder": FE.ENCODER_VERSION,
        "group_ids": sorted({c["group_id"] for c in cases}),
        "template_groups": sorted({c["template_group"] for c in cases}),
        "decide_states": len(decide), "row_states": len(rows)}}


# -- fitting (torch) ------------------------------------------------------------------------
def _assemble(states, device, encoder: str = "tokens"):
    import torch
    seqs, segs, structs, owner, labels = [], [], [], [], []
    for s_index, state in enumerate(states):
        for cand, struct, label in zip(state["cands"], state["structs"], state["labels"]):
            ids = [FE.SPECIALS["cls"]] + state["need"] + [FE.SPECIALS["sep"]] + cand
            seqs.append(ids)
            segs.append(FE.segments(state["need"], cand, encoder))
            structs.append(struct)
            owner.append(s_index)
            labels.append(label)
    width = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), width, dtype=torch.long)
    seg = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, (s, g) in enumerate(zip(seqs, segs)):
        ids[i, :len(s)] = torch.tensor(s)
        seg[i, :len(g)] = torch.tensor(g)
    return (ids.to(device), seg.to(device), torch.tensor(structs, dtype=torch.float32, device=device),
            torch.tensor(owner, device=device), torch.tensor(labels, device=device))


def _listwise_loss(scores, owner, labels, n_states):
    import torch
    neg_inf = torch.finfo(scores.dtype).min
    full = torch.full((n_states,), neg_inf, device=scores.device, dtype=scores.dtype)
    lse_all = torch.zeros(n_states, device=scores.device, dtype=scores.dtype)
    lse_pos = torch.zeros(n_states, device=scores.device, dtype=scores.dtype)
    # logsumexp per state through scatter of max-shifted exponentials
    maxes = full.scatter_reduce(0, owner, scores.detach(), reduce="amax")
    shifted = (scores - maxes[owner]).exp()
    lse_all = lse_all.index_add(0, owner, shifted)
    lse_pos = lse_pos.index_add(0, owner, shifted * labels.float())
    return (lse_all.log() - lse_pos.clamp(min=1e-30).log()).mean()


def _utility_listwise_loss(scores, owner, utilities, n_states):
    """Soft listwise targets from counterfactual packet utility, skipping empty slates."""
    import torch
    utilities = utilities.clamp(min=0)
    sums = torch.zeros(n_states, device=scores.device, dtype=scores.dtype).index_add(0, owner, utilities)
    valid = sums > 0
    if not bool(valid.any()):
        return scores.sum() * 0.0
    neg_inf = torch.finfo(scores.dtype).min
    maxes = torch.full((n_states,), neg_inf, device=scores.device, dtype=scores.dtype)
    maxes = maxes.scatter_reduce(0, owner, scores.detach(), reduce="amax")
    exp_sums = torch.zeros(n_states, device=scores.device, dtype=scores.dtype)
    exp_sums = exp_sums.index_add(0, owner, (scores - maxes[owner]).exp())
    log_probs = scores - maxes[owner] - exp_sums[owner].clamp_min(1e-30).log()
    targets = utilities / sums[owner].clamp_min(1e-30)
    losses = torch.zeros(n_states, device=scores.device, dtype=scores.dtype)
    losses = losses.index_add(0, owner, -targets * log_probs)
    return losses[valid].mean()


def evaluate_ranking(model, states, device, batch=64) -> dict:
    import torch
    model.eval()
    hits = total = utility_hits = utility_states = 0
    selected_utility = 0.0
    with torch.no_grad():
        for i in range(0, len(states), batch):
            chunk = states[i:i + batch]
            ids, seg, structs, owner, labels = _assemble(chunk, device, model.encoder)
            scores, _ = model(ids, seg, structs)
            for s in range(len(chunk)):
                m = owner == s
                best = scores[m].argmax()
                hits += int(labels[m][best].item() == 1)
                total += 1
                if model.utility_heads and "utilities" in chunk[s]:
                    values = torch.tensor(chunk[s]["utilities"], device=device)
                    if float(values.max()) > 0:
                        utility_hits += int(values[best] >= values.max() - 1e-6)
                        selected_utility += float(values[best])
                        utility_states += 1
    model.train()
    result = {"top1_positive": hits / total if total else None, "states": total}
    if model.utility_heads:
        result.update({"utility_top1_optimal": utility_hits / utility_states if utility_states else None,
                       "utility_top1_mean_value": selected_utility / utility_states if utility_states else None,
                       "utility_states": utility_states})
    return result


def fit(config: dict, data: dict, *, epochs: int, lr: float, batch_states: int, device: str, seed: int,
        shuffle_labels: bool = False, dev: dict | None = None, threads: int = 6, log=print,
        max_steps: int | None = None) -> tuple:
    import torch
    from .models import Router, flops_per_candidate, parameter_counts
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    model = Router(config).to(device)
    rng = random.Random(seed)
    decide = [dict(s) for s in data["decide"]]
    if model.utility_heads and any("utilities" not in s for s in decide):
        raise ValueError("dual utility heads require states with private packet-marginal targets")
    if shuffle_labels:
        for s in decide:
            labels = list(s["labels"])
            rng.shuffle(labels)
            s["labels"] = labels
            if model.utility_heads:
                utilities = list(s["utilities"])
                rng.shuffle(utilities)
                s["utilities"] = utilities
    rows = [dict(s) for s in data["rows"]] if model.exclusion_head is not None else []
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(decide) / batch_states)
    total_steps = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / 50) *
                                              0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)))
    curve, started, step = [], time.perf_counter(), 0
    train_flops = 0
    per_cand = flops_per_candidate(model)["flops"]
    for epoch in range(epochs):
        rng.shuffle(decide)
        rng.shuffle(rows)
        row_batch = max(1, len(rows) // max(1, steps_per_epoch)) if rows else 0
        running = []
        for b in range(steps_per_epoch):
            chunk = decide[b * batch_states:(b + 1) * batch_states]
            ids, seg, structs, owner, labels = _assemble(chunk, device, model.encoder)
            scores, _ = model(ids, seg, structs)
            if model.utility_heads:
                values = torch.tensor([v for state in chunk for v in state["utilities"]],
                                      dtype=scores.dtype, device=device).clamp(0, 1)
                loss = (_utility_listwise_loss(scores, owner, values, len(chunk))
                        + 0.2 * torch.nn.functional.smooth_l1_loss(scores.sigmoid(), values))
            else:
                loss = _listwise_loss(scores, owner, labels, len(chunk))
            train_flops += 3 * per_cand * ids.shape[0]
            if rows:
                rchunk = rows[b * row_batch:(b + 1) * row_batch]
                if rchunk:
                    rids, rseg, rstruct, _owner, rlabels = _assemble(rchunk, device, model.encoder)
                    _s, logits = model(rids, rseg, rstruct)
                    loss = loss + 0.5 * torch.nn.functional.cross_entropy(logits, rlabels)
                    train_flops += 3 * per_cand * rids.shape[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running.append(float(loss.detach()))
            if max_steps is not None and step >= max_steps:
                break
            if step % 100 == 0:
                log(f"  step {step}/{total_steps} loss {sum(running[-100:]) / len(running[-100:]):.4f} "
                    f"{time.perf_counter() - started:.0f}s")
        curve.append({"epoch": epoch + 1, "mean_loss": sum(running) / len(running)})
        if max_steps is not None and step >= max_steps:
            break
    receipt = {"config": config, "epochs": epochs, "lr": lr, "batch_states": batch_states, "seed": seed,
               "shuffle_labels": shuffle_labels, "device": device, "threads": threads,
               "train_seconds": time.perf_counter() - started, "steps": step, "loss_curve": curve,
               "train_states": len(decide), "train_row_states": len(rows),
               "objective": "packet-marginal-utility-listwise+brier" if model.utility_heads else "binary-listwise",
               "train_flops_estimate": train_flops,
               "train_flops_method": "3 x measured forward FLOPs per candidate at fixed length x candidates seen "
                                     "(upper bound; batches use shorter dynamic padding)",
               "parameters": parameter_counts(model), "dataset": data["meta"]}
    from .run import peak_rss_bytes
    receipt["memory"] = {"process_peak_rss_bytes": peak_rss_bytes(),
                         "scope": "whole training process, including the in-memory dataset"}
    if device == "cuda":
        receipt["memory"]["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
    if dev is not None:
        receipt["dev_ranking"] = evaluate_ranking(model, dev["decide"], device)
    return model, receipt


def save_checkpoint(model, receipt: dict, path: Path) -> str:
    import torch
    from .models import file_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"config": model.config, "state_dict": model.state_dict(), "receipt": receipt}, tmp)
    os.replace(tmp, path)
    return file_sha256(str(path))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--config", type=Path, required=True)
    d.add_argument("--split", choices=("train", "dev"), required=True)
    d.add_argument("--out", type=Path, required=True)
    tp = sub.add_parser("trainprobe", help="peak memory and step time for a fixed number of training steps")
    tp.add_argument("--config", type=Path, required=True)
    tp.add_argument("--model", required=True)
    tp.add_argument("--train", type=Path, required=True)
    tp.add_argument("--steps", type=int, default=40)
    tp.add_argument("--device", default="cpu")
    f = sub.add_parser("fit")
    f.add_argument("--config", type=Path, required=True)
    f.add_argument("--model", required=True, help="model name from the config's models table")
    f.add_argument("--train", type=Path, required=True)
    f.add_argument("--dev", type=Path)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--device", default="cpu")
    f.add_argument("--cuda-memory-fraction", type=float)
    args = p.parse_args(argv)
    if getattr(args, "cuda_memory_fraction", None) and args.device == "cuda":
        import torch
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.cmd == "dataset":
        spec = cfg["dataset"]
        data = build_dataset(args.split, spec["worlds"][args.split], spec["seed"][args.split], spec["catalog_size"],
                             spec["arm_configs"], spec["epsilons"], spec["max_candidates"], spec["max_tool_calls"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        import gzip
        with gzip.open(args.out, "wt", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
        print(json.dumps(data["meta"], indent=1)[:2000])
        return 0
    import gzip
    if args.cmd == "trainprobe":
        from .run import peak_rss_bytes
        data = json.load(gzip.open(args.train, "rt", encoding="utf-8"))
        spec, train = cfg["models"][args.model], cfg["training"]
        import torch
        torch.zeros(1)
        if args.device == "cuda":
            torch.zeros(1, device="cuda")
            torch.cuda.reset_peak_memory_stats()
        before = peak_rss_bytes()
        started = time.perf_counter()
        model, receipt = fit(spec["config"], data, epochs=1, lr=train["lr"], batch_states=train["batch_states"],
                             device=args.device, seed=train["seed"], threads=train.get("threads", 6),
                             log=lambda *_: None, max_steps=args.steps)
        out = {"model": args.model, "device": args.device, "steps": receipt["steps"],
               "seconds_per_step": (time.perf_counter() - started) / max(1, receipt["steps"]),
               "batch_states": train["batch_states"], "peak_rss_after_dataset_and_torch_load": before,
               "peak_rss_training": peak_rss_bytes(), "parameters": receipt["parameters"]}
        if args.device == "cuda":
            out["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
        print(json.dumps(out))
        return 0
    data = json.load(gzip.open(args.train, "rt", encoding="utf-8"))
    dev = json.load(gzip.open(args.dev, "rt", encoding="utf-8")) if args.dev else None
    spec = cfg["models"][args.model]
    train = cfg["training"]
    model, receipt = fit(spec["config"], data, epochs=spec.get("epochs", train["epochs"]),
                         lr=spec.get("lr", train["lr"]), batch_states=train["batch_states"], device=args.device,
                         seed=spec.get("seed", train["seed"]), shuffle_labels=spec.get("shuffle_labels", False),
                         dev=dev, threads=train.get("threads", 6))
    receipt["model_name"] = args.model
    sha = save_checkpoint(model, receipt, args.out)
    receipt["checkpoint_sha256"] = sha
    args.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("model_name", "train_seconds", "steps", "dev_ranking",
                                              "checkpoint_sha256") if k in receipt}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
