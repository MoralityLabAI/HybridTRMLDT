"""Stage 3: coverage-aware LDT/TRM hybrid on graph skills with noisy or incomplete observations.

Observation model.  The true dependency/dominance graph determines the label.  The
public graph keeps each true edge as *certain* (weight 1) with probability 1-p_drop;
dropped true edges and a number of spurious edges become *uncertain* edges whose
relation weight is a detector score in (0, 1):
``noisy`` strata draw informative scores (sigmoid(N(+1,1)) for real, sigmoid(N(-1,1))
for spurious); ``incomplete`` strata give every uncertain edge the uninformative 0.5.
Spurious edges respect a topological order of the true graph, so every realization
of the uncertain edges is a DAG.

Three-valued certificates (``certify3``) are sound for *every* realization of the
uncertain edges, hence for the true graph:

* retrieval: IN if reachable from the query along certain edges; OUT if unreachable
  even along certain + uncertain edges; otherwise unresolved.
* deduction (kernel of the dominance DAG): ALIVE if every potential dominator
  (certain or uncertain) is certified DEAD; DEAD if some certain dominator is
  certified ALIVE; iterate to a fixpoint (bounded rounds); otherwise unresolved.

Proof obligations: ``soundness_obligation`` enumerates all 2^k realizations for small
k and checks each certified status against the exact solver on every realization.
The campaign records zero-violation receipts for the test strata against hidden
labels as well.

Arms (test, noisy/incomplete strata): exact solver on the thresholded public graph
(uncertain edge present iff score > 0.5), certificates + best constant fill,
neural-only TRM, coverage-aware hybrid (certificate status, depth and unresolved
flag as features; certified slots projected; loss on unresolved slots only), and an
adaptive hybrid policy (symbolic rounds -> exact-solver route when fully resolved ->
neural -> verifier call only when confidence < tau, tau chosen on calibration).
On complete-observation strata the exact solver is routed directly.

All host work is metered as edge examinations; model FLOPs are counted only for
instances the network actually executes; CPU time excludes grading.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from research_gym.trm_skill_gym import campaign as C, dsl, modules as M, tasks as T, train as TR
from . import graphs as G
from . import supervisor as SUP

TRAIN_ROOT, CAL_ROOT, TEST_ROOT = 5_000_000, 5_100_000, 5_200_000
STATUS_COL, DEPTH_COL, UNRES_COL = M.CERT_COL, M.CERT_COL - 1, M.CERT_COL - 2
BASE_FAMILIES = {"retrieval": ["layered_shallow", "layered_wide", "branching", "shared_subdeps", "distractors", "misleading", "long_chain_10"],
                 "deduction": ["kernel_h4", "kernel_h5", "branching", "shared_subdeps", "distractors", "misleading", "long_chain_10"]}
WITHHELD = {"retrieval": ["layered_deep", "ladder"], "deduction": ["kernel_h7", "ladder"]}
MODES = {"noisy": (0.3, 0.2, True), "incomplete": (0.3, 0.2, False)}  # p_drop, spurious fraction, informative


# ------------------------------------------------------------------ observation model
def _topo(rel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Topological position (parents first) of the graph rel[i, j] = 1: i depends on / is dominated by j."""
    n = rel.shape[0]
    indeg = rel.sum(1).astype(int)  # number of parents
    pos = np.full(n, -1)
    queue = [i for i in range(n) if indeg[i] == 0]
    k = 0
    while queue:
        u = queue.pop(0)
        pos[u] = k
        k += 1
        for c in np.nonzero(rel[:, u])[0]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(int(c))
    if (pos < 0).any():
        raise ValueError("graph is not a DAG")
    return pos


def corrupt(skill: str, inst: T.Instance, mode: str, rng: np.random.Generator) -> T.Instance:
    p_drop, spur, informative = MODES[mode]
    rel = inst.rel.copy()
    n = int(inst.mask.sum())
    pos = _topo(rel[:n, :n], inst.mask[:n])
    true_edges = list(zip(*np.nonzero(rel[:n, :n])))
    obs = np.zeros_like(rel)
    for i, j in true_edges:
        if rng.random() < p_drop:
            s = 1 / (1 + np.exp(-rng.normal(1.0, 1.0))) if informative else 0.5
            obs[i, j] = min(max(s, 0.02), 0.98)
        else:
            obs[i, j] = 1.0
    n_spur = int(round(spur * max(len(true_edges), 1)))
    tries = 0
    while n_spur > 0 and tries < 500:
        tries += 1
        i, j = rng.integers(0, n, 2)
        if i == j or rel[i, j] > 0 or obs[i, j] > 0 or pos[j] >= pos[i]:
            continue
        s = 1 / (1 + np.exp(-rng.normal(-1.0, 1.0))) if informative else 0.5
        obs[i, j] = min(max(s, 0.02), 0.98)
        n_spur -= 1
    out = T.Instance(cand=inst.cand.copy(), rel=obs.astype(np.float32), glob=inst.glob.copy(), mask=inst.mask.copy(),
                     label=inst.label.copy(), aux=inst.aux.copy(), auxg=inst.auxg.copy())
    # structural degree features recomputed from the public (possible-edge) graph only
    c = (out.rel > 0).astype(np.float32)
    if skill == "retrieval":
        out.cand[:, 10], out.cand[:, 11] = c.sum(1) / 4, c.sum(0) / 4
    else:
        out.cand[:, 1], out.cand[:, 2] = c.sum(1) / 3, c.sum(0) / 3
        out.cand[:, 3] = (c.sum(1) == 0) * out.mask
    return out


def base_instance(skill: str, family: str, root: int) -> T.Instance:
    inst = G.instance(skill, family, root)
    for c in G.HINT_COLS[skill]:  # hint-free throughout this stage
        inst.cand[:, c] = 0.5 * inst.mask
    return inst


def make(skill: str, stratum: str, root: int) -> T.Instance:
    """stratum = '<mode>:<family>' with mode in noisy|incomplete|complete."""
    mode, family = stratum.split(":", 1)
    inst = base_instance(skill, family, root)
    if mode == "complete":
        return inst
    rng = np.random.default_rng([int(root), sum(map(ord, stratum)), 77])
    return corrupt(skill, inst, mode, rng)


def strata(skill: str, stage: str) -> List[str]:
    fams = BASE_FAMILIES[skill] + (WITHHELD[skill] if stage == "test" else [])
    return [f"{m}:{f}" for m in ("noisy", "incomplete", "complete") for f in fams]


# ------------------------------------------------------------------ certificates (public graph only)
def certify3(skill: str, rel: np.ndarray, mask: np.ndarray, query: int | None, max_rounds: int = 32) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (status in {+1,-1,0}, resolution round, edge examinations). Uses only rel/mask/query."""
    n = int(mask.sum())
    certain = rel[:n, :n] >= 1.0
    possible = rel[:n, :n] > 0
    st = np.zeros(n, int)
    depth = np.zeros(n, int)
    work = 0
    if skill == "retrieval":
        reach_c = np.zeros(n, bool)
        reach_p = np.zeros(n, bool)
        frontier_c, frontier_p = {query}, {query}
        for r in range(1, max_rounds + 1):
            nc = {int(j) for i in frontier_c for j in np.nonzero(certain[i])[0]} - set(np.nonzero(reach_c)[0]) - {query}
            np_ = {int(j) for i in frontier_p for j in np.nonzero(possible[i])[0]} - set(np.nonzero(reach_p)[0]) - {query}
            work += sum(int(possible[i].sum()) for i in frontier_p) + sum(int(certain[i].sum()) for i in frontier_c)
            for j in nc:
                reach_c[j] = True
                depth[j] = r
            for j in np_:
                reach_p[j] = True
            frontier_c, frontier_p = nc, np_
            if not nc and not np_:
                break
        st[reach_c] = 1
        out = ~reach_p
        out[query] = True
        st[out] = -1
        depth[out & (depth == 0)] = max_rounds if frontier_p else r
        return st, depth, work
    # deduction
    for r in range(1, max_rounds + 1):
        nxt = st.copy()
        for i in range(n):
            if st[i] != 0:
                continue
            doms_p = np.nonzero(possible[i])[0]
            doms_c = np.nonzero(certain[i])[0]
            work += len(doms_p)
            if any(st[j] == 1 for j in doms_c):
                nxt[i] = -1
                depth[i] = r
            elif all(st[j] == -1 for j in doms_p):
                nxt[i] = 1
                depth[i] = r
        if (nxt == st).all():
            break
        st = nxt
    return st, depth, work


def exact_solve(skill: str, adj: np.ndarray, query: int | None) -> np.ndarray:
    """Exact solver on a fully known graph (adj boolean)."""
    n = adj.shape[0]
    if skill == "retrieval":
        seen = np.zeros(n, bool)
        stack = [query]
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        seen[query] = False
        return seen.astype(np.float32)
    pos = _topo(adj.astype(np.float32), np.ones(n))
    alive = np.zeros(n, bool)
    for u in np.argsort(pos):
        alive[u] = not any(alive[j] for j in np.nonzero(adj[u])[0])
    return alive.astype(np.float32)


def soundness_obligation(skill: str, inst: T.Instance, max_uncertain: int = 10) -> Dict:
    """Enumerate every realization of the uncertain edges and check certified slots."""
    n = int(inst.mask.sum())
    rel = inst.rel[:n, :n]
    unc = list(zip(*np.nonzero((rel > 0) & (rel < 1))))
    if len(unc) > max_uncertain:
        return {"checked": False, "uncertain_edges": len(unc)}
    q = int(np.argmax(inst.aux[:n, 0])) if skill == "retrieval" else None
    st, _, _ = certify3(skill, inst.rel, inst.mask, q)
    violations = 0
    for bits in itertools.product((0, 1), repeat=len(unc)):
        adj = rel >= 1.0
        adj = adj.copy()
        for (i, j), b in zip(unc, bits):
            adj[i, j] = bool(b)
        sol = exact_solve(skill, adj, q)
        violations += int(((st == 1) & (sol < 0.5)).sum() + ((st == -1) & (sol > 0.5)).sum())
    return {"checked": True, "uncertain_edges": len(unc), "realizations": 2 ** len(unc), "violations": violations}


def query_of(b: Dict[str, torch.Tensor], i: int) -> int:
    return int(torch.argmax(b["cand"][i, :, 0]))


def annotate(skill: str, b: Dict[str, torch.Tensor], rounds: int = 32) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
    """Write certificate status/depth/unresolved into reserved public columns."""
    out = {k: v.clone() for k, v in b.items()}
    cert = torch.zeros_like(b["mask"])
    work = 0
    for i in range(len(b["mask"])):
        q = query_of(b, i) if skill == "retrieval" else None
        st, depth, w = certify3(skill, b["rel"][i].numpy(), b["mask"][i].numpy(), q, rounds)
        work += w
        n = len(st)
        cert[i, :n] = torch.from_numpy(st).float()
        out["cand"][i, :n, DEPTH_COL] = torch.from_numpy(depth).float() / 16
    out["cand"][..., STATUS_COL] = cert
    out["cand"][..., UNRES_COL] = ((cert == 0) & (b["mask"] > 0)).float()
    return out, cert, work


def pool(skill: str, stage: str, count: int) -> Dict[str, Dict[str, torch.Tensor]]:
    base = CAL_ROOT if stage == "cal" else TEST_ROOT
    return {s: T.stack([make(skill, s, base + 10_000 * k + i) for i in range(count)]) for k, s in enumerate(strata(skill, stage))}


def train_pool(skill: str, count: int) -> Dict[str, torch.Tensor]:
    ss = [s for s in strata(skill, "cal") if not s.startswith("complete")]
    return T.stack([make(skill, ss[i % len(ss)], TRAIN_ROOT + i) for i in range(count)])


# ------------------------------------------------------------------ training
def train_hybrid(skill: str, steps: int, seed: int, train: Dict[str, torch.Tensor], coverage: bool):
    """Neural-only: plain TRM training.  Coverage-aware: certificate columns are public
    inputs and certified logits are projected (clamped) before the loss, so certified
    slots carry no gradient and learning concentrates on unresolved slots."""
    cfg = dsl.validate({"core": "trm", "ldt_projection": False, "verifier_visits": 0})
    if not coverage:
        model, info = TR.train_module(skill, cfg, steps, seed, train)
        return model, cfg, info
    train, cert, _ = annotate(skill, train)
    torch.manual_seed(seed)
    model = M.SkillModule(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    n = train["cand"].shape[0]
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n, (TR.BATCH,), generator=g)
        b = TR.take(train, idx)
        out = M.episode(model, skill, b, cfg)
        out["logits"] = [M.project(l, cert[idx]) for l in out["logits"]]
        out["pred"] = [T.decide(skill, l.detach(), b["mask"]) for l in out["logits"]]
        out["exact"] = [T.exact(skill, b, pr) for pr in out["pred"]]
        loss = M.episode_loss(skill, out, b, cfg, b["label"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return model, cfg, {"steps": len(losses), "final_loss": float(np.mean(losses[-20:])) if losses else None}


@torch.no_grad()
def neural_predict(model, cfg, skill, pub) -> Tuple[torch.Tensor, torch.Tensor]:
    cand = pub["cand"].clone()
    cand[..., T.FS:] = 0
    st = model.init_state(pub)
    for _ in range(cfg["visits"]):
        logits, conf, value, st = model.visit(cand, pub, st, cfg["visits"])
    return T.decide(skill, logits, pub["mask"]), torch.sigmoid(conf)


@torch.no_grad()
def run_arm(arm: str, skill: str, b: Dict[str, torch.Tensor], models: Dict, tau: float = 0.9, const: float = 0.0) -> Dict:
    t0 = time.process_time()
    B = len(b["mask"])
    work = 0
    flops = 0
    neural_n = 0
    vcalls = 0
    pred = torch.zeros_like(b["mask"])
    cert = torch.zeros_like(b["mask"])
    if arm == "threshold_exact":
        for i in range(B):
            n = int(b["mask"][i].sum())
            adj = b["rel"][i, :n, :n].numpy() > 0.5
            q = query_of(b, i) if skill == "retrieval" else None
            work += int((b["rel"][i] > 0).sum())
            try:
                pred[i, :n] = torch.from_numpy(exact_solve(skill, adj, q))
            except ValueError:
                pass
    elif arm == "neural_only":
        M.flops_reset()
        pred, _ = neural_predict(models["neural"], models["cfg"], skill, b)
        flops = M.flops_read()
        neural_n = B
    else:
        ann, cert, work = annotate(skill, b)
        unresolved = ((cert == 0) & (b["mask"] > 0)).any(1)
        pred = (cert > 0).float()
        if arm == "cert_const":
            pred = torch.where((cert == 0) & (b["mask"] > 0), torch.full_like(pred, const), pred)
        elif arm in ("hybrid", "hybrid_adaptive") and unresolved.any():
            idx = torch.nonzero(unresolved).squeeze(1)
            pub = {k: ann[k][idx] for k in ("cand", "rel", "glob", "mask")}
            M.flops_reset()
            st = models["hybrid"].init_state(pub)
            cfgh = models["cfg_h"]
            for _ in range(cfgh["visits"]):
                logits, conf, value, st = models["hybrid"].visit(cand_with_cert(pub["cand"]), pub, st, cfgh["visits"])
            logits = M.project(logits, cert[idx])
            proposed = T.decide(skill, logits, pub["mask"])
            flops = M.flops_read()
            neural_n = len(idx)
            if arm == "hybrid_adaptive":
                low = torch.sigmoid(conf) < tau
                if low.any():  # one verifier call; on rejection fall back to the thresholded exact solver
                    sub = {k: b[k][idx[low]] for k in b}
                    ok, _ = T.verify(skill, sub, proposed[low])
                    vcalls += int(low.sum())
                    work += int(low.sum()) * T.N * T.N
                    fb = run_arm("threshold_exact", skill, sub, models)
                    work += fb["host_edge_examinations"]
                    proposed[low] = torch.where(ok[:, None], proposed[low], fb["_pred"])
            pred[idx] = proposed
    cpu = time.process_time() - t0
    acc = T.exact(skill, b, pred * b["mask"])
    unresolved_inst = ((cert == 0) & (b["mask"] > 0)).any(1) if arm not in ("threshold_exact", "neural_only") else None
    viol = int((((cert > 0) & (b["label"] < 0.5)) | ((cert < 0) & (b["label"] > 0.5))).sum())
    return {"exact": float(acc.mean()), "per_root": acc.numpy().astype(np.int8).tolist(),
            "coverage": float(((cert != 0).float() * b["mask"]).sum() / b["mask"].sum()) if arm not in ("threshold_exact", "neural_only") else None,
            "unresolved_instances": None if unresolved_inst is None else int(unresolved_inst.sum()),
            "exact_on_unresolved": None if unresolved_inst is None or not unresolved_inst.any() else float(acc[unresolved_inst].mean()),
            "certificate_violations": viol, "host_edge_examinations": work, "model_flops": flops,
            "neural_instances": neural_n, "verifier_calls": vcalls, "cpu_s": cpu, "_pred": pred}


def cand_with_cert(c: torch.Tensor) -> torch.Tensor:
    """Keep the reserved certificate columns, clear the verifier feedback channel."""
    out = c.clone()
    out[..., T.FS:] = 0
    return out


# ------------------------------------------------------------------ worker / driver
def worker(spec: Dict) -> Dict:
    torch.set_num_threads(1)
    skill, seed, steps = spec["skill"], spec["seed"], spec["steps"]
    train = train_pool(skill, spec["train_count"])
    t0 = time.process_time()
    neural, cfg, info_n = train_hybrid(skill, steps, seed, train, coverage=False)
    t1 = time.process_time()
    hyb, cfg_h, info_h = train_hybrid(skill, steps, seed, train, coverage=True)
    t2 = time.process_time()
    complete = info_n["steps"] == steps and info_h["steps"] == steps
    models = {"neural": neural.eval(), "cfg": cfg, "hybrid": hyb.eval(), "cfg_h": cfg_h}
    cal = pool(skill, "cal", spec["count"])
    # calibration choices: constant fill and adaptive tau
    res_cal = {s: run_arm("cert_const", skill, b, models, const=0.0)["exact"] for s, b in cal.items()}
    res_cal1 = {s: run_arm("cert_const", skill, b, models, const=1.0)["exact"] for s, b in cal.items()}
    const = 0.0 if np.mean(list(res_cal.values())) >= np.mean(list(res_cal1.values())) else 1.0
    taus = [0.5, 0.7, 0.9, 0.97]
    tau_score = {}
    for tau in taus:
        rs = [run_arm("hybrid_adaptive", skill, b, models, tau=tau) for b in cal.values()]
        tau_score[tau] = float(np.mean([r["exact"] for r in rs]) - 0.002 * np.mean([r["verifier_calls"] for r in rs]))
    tau = max(tau_score, key=tau_score.get)
    frozen = {"const": const, "tau": tau, "tau_score": tau_score}
    test = pool(skill, "test", spec["test_count"])
    arms = ("threshold_exact", "cert_const", "neural_only", "hybrid", "hybrid_adaptive")
    out = {}
    for s, b in test.items():
        out[s] = {}
        for arm in arms:
            r = run_arm(arm, skill, b, models, tau=tau, const=const)
            r.pop("_pred")
            out[s][arm] = r
    obligations = []
    for s in strata(skill, "test"):
        if s.startswith("complete"):
            continue
        for i in range(8):
            inst = make(skill, s, TEST_ROOT + 900_000 + i)
            ob = soundness_obligation(skill, inst)
            ob["stratum"] = s
            obligations.append(ob)
    return {"status": "ok" if complete else "incomplete", "train": {"neural": info_n, "hybrid": info_h,
            "neural_cpu_s": t1 - t0, "hybrid_cpu_s": t2 - t1}, "frozen_on_cal": frozen, "test": out,
            "proof_obligations": obligations}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--skills", nargs="+", default=["retrieval", "deduction"])
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11])
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--test-count", type=int, default=256)
    p.add_argument("--train-count", type=int, default=4096)
    p.add_argument("--hours", type=float, default=1.0)
    a = p.parse_args(argv)
    if a.worker:
        spec = json.loads(a.worker.read_text(encoding="utf-8"))
        SUP.write_json(Path(spec["out"]), worker(spec))
        return
    if not 0 < a.hours <= 8:
        p.error("hours in (0, 8]")
    a.out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 3600 * a.hours
    src = {f.name: hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in sorted(Path(__file__).parent.glob("*.py"))}
    SUP.write_json(a.out / "manifest.json", {"args": {k: str(v) for k, v in vars(a).items()}, "source_sha256": src,
                                             "roots": {"train": TRAIN_ROOT, "cal": CAL_ROOT, "test": TEST_ROOT},
                                             "strata": {s: strata(s, "test") for s in a.skills}})
    results = {}
    for skill in a.skills:
        for seed in a.seeds:
            out = a.out / "jobs" / f"{skill}__s{seed}.json"
            if not out.exists():
                if time.time() > deadline:
                    results[f"{skill}/s{seed}"] = {"status": "deadline_not_run"}
                    continue
                spec = {"skill": skill, "seed": seed, "steps": a.steps, "count": a.count, "test_count": a.test_count,
                        "train_count": a.train_count, "out": str(out)}
                sp = a.out / "specs" / f"{skill}__s{seed}.json"
                SUP.write_json(sp, spec)
                sup = SUP.run(SUP.python_module("research_gym.trm_skill_gym_v3.hybrid", ["--worker", str(sp)]),
                              wall_s=3 * 3600, log_path=a.out / "workers.log")
                if not out.exists():
                    SUP.write_json(out, {"status": sup["status"], "supervisor": sup})
                else:
                    r = json.loads(out.read_text(encoding="utf-8"))
                    r["supervisor"] = sup
                    SUP.write_json(out, r)
            results[f"{skill}/s{seed}"] = json.loads(out.read_text(encoding="utf-8"))
            print(skill, seed, results[f"{skill}/s{seed}"]["status"], flush=True)
    SUP.write_json(a.out / "results.json", results)


if __name__ == "__main__":
    main()
