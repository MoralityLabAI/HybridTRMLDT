"""Audit of the MOLT-GOBLIN-02 finding: reproduction, causality, parameters, visits, loop forcing.

1. Re-run frozen MOLT-02 hidden-root cells with the MOLT-02 code and compare with the
   committed final_runs records (exact and per-root outcomes).
2. Causality of every MOLT-02 finalist: perturbing token t never changes outputs < t.
3. Parameter accounting by component.
4. Module visits and measured forward FLOPs (FlopCounterMode).
5. Loop forcing: which loops' readouts receive gradient under each schedule.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from research_gym.molt_goblin_02 import finals as f2mod
from research_gym.molt_goblin_02.factory import MoltConfig, MoltNet, loss_fn as loss_v2
from research_gym.molt_goblin_02.gym import FAMILIES, TaskSpec
from research_gym.molt_goblin_02.train import HIDDEN_BASE, train_and_eval

ROOT = Path(__file__).resolve().parents[1] / "molt_goblin_02" / "receipts" / "finals_20261128"


def recorded() -> dict:
    out = {}
    for p in sorted(ROOT.glob("final_runs.shard*.jsonl.gz")):
        for line in gzip.open(p, "rt", encoding="utf-8"):
            r = json.loads(line)
            out[r["key"]] = r
    return out


def reproduce(cells) -> list[dict]:
    frozen = json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))
    rec = recorded()
    rows = []
    for arm, family, seed in cells:
        cfg = MoltConfig.from_dict(frozen["arms"][arm]["config"])
        p = f2mod.ALL[family]
        r = train_and_eval(cfg, family, p["train"], p["eval"], seed, n_eval=256, base=HIDDEN_BASE,
                           extrap_loops=min(12, 2 * cfg.loops))
        old = rec[f"{arm}|{family}|{seed}"]
        rows.append({"arm": arm, "family": family, "seed": seed, "exact_now": r["evals"]["dev"]["exact"],
                     "exact_recorded": old["evals"]["dev"]["exact"],
                     "per_root_identical": r["evals"]["dev"]["per_example"] == old["evals"]["dev"]["per_example"],
                     "train_cpu_s_now": r["train_cpu_s"], "train_cpu_s_recorded": old["train_cpu_s"]})
        print(json.dumps(rows[-1]), flush=True)
    return rows


@torch.no_grad()
def causality(cfg: MoltConfig, family: str) -> float:
    fam = FAMILIES[family]
    torch.manual_seed(0)
    net = MoltNet(cfg, fam.vocab, fam.n_out).eval()
    x = fam.generate(f2mod.ALL[family]["train"], 16, 3).x
    worst = 0.0
    for t in range(1, x.shape[1]):
        x2 = x.clone()
        x2[:, t] = x[torch.randperm(16), t]
        a, b = net(x)["logits"][-1], net(x2)["logits"][-1]
        worst = max(worst, float((a[:, :t] - b[:, :t]).abs().max()))
    return worst


def components(net: torch.nn.Module) -> dict:
    out = {}
    for name, p in net.named_parameters():
        key = name.split(".")[0]
        out[key] = out.get(key, 0) + p.numel()
    return out


def flops(net, x) -> float:
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(x)
    return fc.get_total_flops() / x.shape[0]


def loop_forcing_gradients(cfg: MoltConfig, family: str) -> dict:
    fam = FAMILIES[family]
    torch.manual_seed(0)
    net = MoltNet(cfg, fam.vocab, fam.n_out)
    b = fam.generate(f2mod.ALL[family]["train"], 16, 3)
    out = net(b.x)
    for lg in out["logits"]:
        lg.retain_grad()
    loss, _ = loss_v2(cfg, out, b.y, b.mask.float())
    loss.backward()
    return {"loss": cfg.loss, "readout_grad_norm_by_loop": [0.0 if lg.grad is None else float(lg.grad.norm())
                                                            for lg in out["logits"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-reproduce", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    frozen = json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))
    arms = {k: MoltConfig.from_dict(v["config"]) for k, v in frozen["arms"].items()}
    fam = "proof_deps"
    x = FAMILIES[fam].generate(f2mod.ALL[fam]["train"], 8, 1).x
    acc = {}
    for k, cfg in arms.items():
        torch.manual_seed(0)
        net = MoltNet(cfg, FAMILIES[fam].vocab, FAMILIES[fam].n_out)
        acc[k] = {"params": sum(p.numel() for p in net.parameters()), "by_component": components(net),
                  "flops_per_example": flops(net, x),
                  "logical_visits": cfg.loops * (cfg.n_types if cfg.routing in ("mix", "hrm") else 1),
                  "max_causal_leak": causality(cfg, fam)}
    f2 = arms["f2_af49a1"]
    lf = {"all": loop_forcing_gradients(f2, fam), "last": loop_forcing_gradients(f2.replace(loss="last"), fam)}
    res = {"accounting_proof_deps": acc, "loop_forcing": lf}
    if not a.skip_reproduce:
        cells = [(arm, fam_, s) for arm in ("f2_af49a1", "prefix") for fam_ in ("proof_deps", "experiment_select")
                 for s in (11, 23)]
        res["reproduction"] = reproduce(cells)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "audit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_component"} for k, v in acc.items()}, indent=0))
    print(json.dumps(lf))


if __name__ == "__main__":
    main()
