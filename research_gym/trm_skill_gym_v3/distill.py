"""GoblinBench architecture distillation, v3.

Strict pre/post record builder, small CPU students (pairwise logistic ranker, ridge,
GRU research controller), heuristic baselines, three holdout schemes, grouped
bootstrap, and DataGoblin-shaped export. Every number is out-of-fold.

Inputs hold only what exists before the experiment: SKILL objective, incumbent,
the proposal, its rationale, Inky's own forecast, resource estimates computed from
the config, the falsification criterion, and diagnoses/results of strictly earlier
rounds of the same arm. Rung results, decision, status, and the proposal's own
diagnosis live under `target` and are never read when building features.

CLI: python -m research_gym.trm_skill_gym_v3.distill --out DIR [--extra-archive PATH ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

try:  # tiny matrices: single-thread BLAS avoids oversubscription on a loaded machine
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except Exception:
    pass

from research_gym.trm_skill_gym import dsl

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPO / "experiments/trm_skill_gym_01/run/search/archive.jsonl"
RUNGS = ("rung0", "rung1")
SKILLS = ("repair", "retrieval", "deduction", "research", "curation", "stopping")
PILOT_TEST_SKILLS = ("research", "stopping")
KEYWORDS = ("saturat", "ceiling", "risk", "simplif", "reduc", "rlt", "damp", "verifier",
            "capacity", "regular")
FIELDS = sorted(dsl.SCHEMA)
SEARCH_FIELDS = ("visits", "verifier_visits")


# --------------------------------------------------------------------------- records
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path) -> str:
    """LF-normalized so CRLF checkouts hash identically."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def row_sha256(row) -> str:
    return sha256_bytes(json.dumps(row, sort_keys=True).encode())


def load_archive(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _norm_delta(key, old, new):
    kind, spec, default = dsl.SCHEMA[key]
    if kind in ("int", "float"):
        old = default if old is None else old
        return abs(float(new) - float(old)) / (spec[1] - spec[0])
    return float(new != old)


def edit_profile(change, incumbent):
    deltas = {k: _norm_delta(k, (incumbent or {}).get(k), v)
              for k, v in (change or {}).items() if k in dsl.SCHEMA}
    return deltas


def primary_field(change, incumbent):
    d = edit_profile(change, incumbent)
    if not d:
        return "none"
    return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


_PARAM_CACHE = {}


def resource_estimate(cfg):
    """Params from SkillModule(cfg); FLOPs are a proxy 2*params*visits*cycles*(latent+1)."""
    if not cfg:
        return {"n_params": None, "flops_proxy": None}
    key = json.dumps(cfg, sort_keys=True)
    if key not in _PARAM_CACHE:
        try:
            from research_gym.trm_skill_gym.modules import SkillModule
            n = int(SkillModule(cfg).n_params())
            f = 2.0 * n * cfg["visits"] * cfg["cycles"] * (cfg["latent_steps"] + 1)
            _PARAM_CACHE[key] = {"n_params": n, "flops_proxy": float(f)}
        except Exception:
            _PARAM_CACHE[key] = {"n_params": None, "flops_proxy": None}
    return dict(_PARAM_CACHE[key])


def _objective(skill):
    from research_gym.trm_skill_gym.skill_templates import TEMPLATES
    t = TEMPLATES[skill]
    return {"title": t.get("title"), "objective": t.get("objective")}


def _dev(m):
    v = (m or {}).get("dev_exact")
    return None if v is None else float(v)


def _find_records(archive_path):
    p = Path(archive_path).resolve()
    for parent in list(p.parents)[:4]:
        c = parent / "goblinbench_records.jsonl"
        if c.exists():
            return c
    return None


def build_records(archives):
    """archives: list of (label, path). Returns records in archive order."""
    items = []
    for label, path in archives:
        src_sha = file_sha256(path)
        rec_path = _find_records(path)
        gb = {}
        if rec_path is not None:
            for r in load_archive(rec_path):
                gb[r["candidate_id"].split("/", 1)[-1]] = r
        for row in load_archive(path):
            items.append((label, str(path), src_sha, gb, row))
    seen = set()
    for label, _, _, _, r in items:
        key = (label, r["arm"], r["round"], r["proposal_index"], r["skill"])
        if key in seen:
            raise ValueError(f"duplicate proposal identity {key}")
        seen.add(key)
    # chronological order within (source, arm)
    order = {}
    for i, (label, _, _, _, r) in enumerate(items):
        order.setdefault((label, r["arm"]), []).append((r["round"], r["proposal_index"], i))
    nxt = {}
    for seq in order.values():
        seq.sort()
        for a, b in zip(seq, seq[1:]):
            nxt[a[2]] = b[2]
    records = []
    for i, (label, path, src_sha, gb, r) in enumerate(items):
        prior = []
        for rr in items:
            if rr[0] == label and rr[4]["arm"] == r["arm"] and rr[4]["round"] < r["round"]:
                prior.append(rr[4])
        prior.sort(key=lambda q: (q["round"], q["proposal_index"]))
        modes, seen_m = [], set()
        for q in prior:
            for field in ("failure_diagnosis", "round_diagnosis"):
                t = q.get(field)
                if t and (q["round"], field) not in seen_m:
                    seen_m.add((q["round"], field))
                    modes.append({"round": q["round"], "skill": q["skill"], "kind": field, "text": t})
        history = [{"round": q["round"], "skill": q["skill"], "proposal_index": q["proposal_index"],
                    "change": q.get("change"), "rung0_dev_exact": _dev(q.get("rung0")),
                    "rung1_dev_exact": _dev(q.get("rung1")), "decision": q.get("decision")}
                   for q in prior]
        res = resource_estimate(r.get("config"))
        inc_res = resource_estimate(r.get("incumbent"))
        nr = items[nxt[i]][4] if i in nxt else None
        key = f"{r['arm']}/r{r['round']}/p{r['proposal_index']}/{r.get('config_id')}"
        g = gb.get(key)
        inp = {
            "task_objective": {"skill": r["skill"], **_objective(r["skill"])},
            "skill": r["skill"], "family": r["family"], "arm": r["arm"], "round": r["round"],
            "proposal_index": r["proposal_index"], "source": label,
            "current_architecture": r["incumbent"],
            "incumbent_dev_exact": r.get("incumbent_dev_exact"),
            "known_failure_modes": modes,
            "prior_results": history,
            "proposed_intervention": {"change": r.get("change") or {}, "config": r.get("config"),
                                      "raw_config": r.get("raw_config"),
                                      "edit_budget": r.get("edit_budget")},
            "mathematical_rationale": r.get("rationale"),
            "expected_effect": {"predicted_dev_exact": r.get("predicted_dev_exact")},
            "predicted_resource_use": {**res, "incumbent_n_params": inc_res["n_params"]},
            "falsification_criterion": r.get("falsification"),
        }
        target = {
            "status": r.get("status"), "decision": r.get("decision"),
            "rung0": r.get("rung0"), "rung1": r.get("rung1"),
            "rung0_dev_exact": _dev(r.get("rung0")), "rung1_dev_exact": _dev(r.get("rung1")),
            "measured_dev_exact": r.get("measured_dev_exact"),
            "diagnosis": {"failure_diagnosis": r.get("failure_diagnosis"),
                          "round_diagnosis": r.get("round_diagnosis")},
            "next_experiment": None if nr is None else {
                "skill": nr["skill"], "round": nr["round"], "proposal_index": nr["proposal_index"],
                "change": nr.get("change"), "rationale": nr.get("rationale")},
            "verification_receipt": {"source_row_sha256": row_sha256(r), "source_file": path,
                                     "source_file_sha256_lf": src_sha},
        }
        records.append({
            "id": f"{label}/{r['arm']}/{r['skill']}/r{r['round']}/p{r['proposal_index']}",
            "group": f"{label}/{r['arm']}/r{r['round']}",
            "skill": r["skill"], "arm": r["arm"],
            "intervention_primary_field": primary_field(r.get("change"), r.get("incumbent")),
            "input": inp, "target": target,
            "export_meta": {
                "candidate_id": g["candidate_id"] if g else f"{label}/{key}",
                "architecture_family": g["architecture_family"] if g else "native_trm_update",
                "dataset_manifest_sha256": g["dataset_manifest_sha256"] if g else src_sha,
                "training_split_hashes": g["training_split_hashes"] if g else {"train": src_sha},
                "manifest_source": "goblinbench_records" if g else "archive_file_sha256_fallback"},
        })
    return records


# --------------------------------------------------------------------------- features
def encode_config(cfg):
    v = []
    for key, (kind, spec, default) in dsl.SCHEMA.items():
        x = (cfg or {}).get(key, default)
        if kind == "enum":
            v.extend(float(x == a) for a in spec)
        elif kind == "bool":
            v.append(float(bool(x)))
        else:
            v.append((float(x) - spec[0]) / (spec[1] - spec[0]))
    return np.array(v)


def _pilot_features(inp):
    inc = encode_config(inp["current_architecture"])
    cfg = inp["proposed_intervention"]["config"]
    prop = encode_config(cfg or inp["current_architecture"])
    prior = inp["incumbent_dev_exact"]
    return np.r_[1, inc, prop - inc, float(prior or 0), float(prior is not None)]


def _compact_features(inp):
    iv = inp["proposed_intervention"]
    inc_cfg = inp["current_architecture"]
    cfg = iv["config"] or inc_cfg
    delta = encode_config(cfg) - encode_config(inc_cfg)
    prof = edit_profile(iv["change"], inc_cfg)
    changed = np.array([float(f in prof) for f in FIELDS])
    prior = inp["incumbent_dev_exact"]
    prior = 0.0 if prior is None else float(prior)
    pred = inp["expected_effect"]["predicted_dev_exact"]
    res = inp["predicted_resource_use"]
    n_par = res["n_params"] or res["incumbent_n_params"] or 1.0
    flops = res["flops_proxy"] or 1.0
    text = (inp["mathematical_rationale"] or "").lower()
    kw = np.array([float(k in text) for k in KEYWORDS])
    return np.r_[delta, sum(prof.values()), len(prof), changed, prior,
                 0.0 if pred is None else float(pred) - prior, float(pred is not None),
                 math.log(n_par), math.log(flops), math.log1p(len(text)), kw]


def _full_extra(inp):
    skill = np.array([float(inp["skill"] == s) for s in SKILLS])
    return np.r_[encode_config(inp["current_architecture"]), skill,
                 float(len(inp["known_failure_modes"])), float(inp["arm"] == "random_mutation")]


def feature_matrices(records):
    P = np.stack([_pilot_features(r["input"]) for r in records])
    C = np.stack([_compact_features(r["input"]) for r in records])
    F = np.hstack([C, np.stack([_full_extra(r["input"]) for r in records])])
    return {"pilot": P, "compact": C, "full": F}


# --------------------------------------------------------------------------- models
def _standardize(xtr, xte, floor=0.05):
    mu = xtr.mean(0)
    sd = np.maximum(xtr.std(0), floor)
    return (xtr - mu) / sd, (xte - mu) / sd


def ridge_fit_predict(x, y, xt, alpha, floor=0.05):
    x, xt = _standardize(x, xt, floor)
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    b = vt.T @ ((sv / (sv ** 2 + alpha)) * (u.T @ (y - y.mean())))
    return xt @ b + y.mean()


def ridge_pilot_predict(x, y, xt):
    """Identical to research_gym.trm_skill_gym_v2.student.fit_predict."""
    return np.clip(ridge_fit_predict(x, y, xt, 10.0), 0, 1)


def ridge_multi(x, y, xt, alphas, floor=0.05):
    """Ridge predictions for several alphas from one SVD."""
    x, xt = _standardize(x, xt, floor)
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    proj = u.T @ (y - y.mean())
    return [xt @ (vt.T @ ((sv / (sv ** 2 + a)) * proj)) + y.mean() for a in alphas]


def ridge_cv_predict(x, y, xt, gid, alphas=(1.0, 10.0, 100.0, 1000.0)):
    """Alpha picked by leave-one-round-out CV inside the training fold only."""
    ug = np.unique(gid)
    err = np.zeros(len(alphas))
    used = 0
    if len(ug) >= 3:
        for g in ug:
            te = gid == g
            if (~te).sum() < 3:
                continue
            ps = ridge_multi(x[~te], y[~te], x[te], alphas)
            err += np.array([np.abs(p - y[te]).sum() for p in ps])
            used += te.sum()
    best = alphas[int(np.argmin(err))] if used else alphas[-1]
    return np.clip(ridge_fit_predict(x, y, xt, best), 0, 1)


def pairwise_fit(x, y, gid, lam=3.0):
    """Logistic regression on within-round feature differences (no intercept)."""
    mu, sd = x.mean(0), np.maximum(x.std(0), 0.05)
    z = (x - mu) / sd
    d, lab = [], []
    for g in np.unique(gid):
        idx = np.where(gid == g)[0]
        for a in idx:
            for b in idx:
                if a < b and y[a] != y[b]:
                    s = 1.0 if y[a] > y[b] else 0.0
                    d.append(z[a] - z[b]); lab.append(s)
                    d.append(z[b] - z[a]); lab.append(1 - s)
    if not d:
        return None
    d, lab = np.array(d), np.array(lab)
    w = np.zeros(d.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(d @ w, -30, 30)))
        grad = d.T @ (p - lab) + lam * w
        H = (d * (p * (1 - p))[:, None]).T @ d + lam * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def pairwise_score(model, x):
    if model is None:
        return np.zeros(len(x))
    mu, sd, w = model
    return ((x - mu) / sd) @ w


def _sequences(records):
    seqs = {}
    for i, r in enumerate(records):
        seqs.setdefault((r["input"]["source"], r["arm"]), {}).setdefault(r["input"]["round"], []).append(i)
    out = []
    for k in sorted(seqs):
        out.append([sorted(seqs[k][rd]) for rd in sorted(seqs[k])])
    return out


def gru_predict(C, y0, y1, inc, seqs, train, visible, hidden=12, epochs=100, seeds=(0, 1)):
    """GRU over each arm's chronological rounds. The state before round t has only seen
    rounds < t; per-proposal head sees (state, proposal features). Returns (pred0, pred1)."""
    import torch
    torch.set_num_threads(1)
    N = len(C)
    tr = np.zeros(N, bool); tr[train] = True
    xz, _ = _standardize(C[tr], C[tr], 0.05)
    mu, sd = C[tr].mean(0), np.maximum(C[tr].std(0), 0.05)
    X = (C - mu) / sd
    obs0 = ~np.isnan(y0); obs1 = ~np.isnan(y1)
    m0 = float(np.mean(y0[tr & obs0])) if (tr & obs0).any() else 0.0
    B, T = len(seqs), max(len(s) for s in seqs)
    Pm = max(len(g) for s in seqs for g in s)
    D = X.shape[1]
    idx = -np.ones((B, T, Pm), int)
    for b, s in enumerate(seqs):
        for t, g in enumerate(s):
            idx[b, t, :len(g)] = g
    pm = idx >= 0
    safe = np.where(pm, idx, 0)
    Xs = X[safe] * pm[..., None]
    vis = visible[safe] & pm
    o0 = (obs0[safe] & vis); o1 = (obs1[safe] & vis)
    r0 = np.where(o0, np.nan_to_num(y0[safe]) - m0, 0.0)
    r1 = np.where(o1, np.nan_to_num(y1[safe]) - inc[safe], 0.0)
    n = np.maximum(pm.sum(2, keepdims=True), 1)
    summ = np.concatenate([Xs.sum(2) / n, r0.sum(2, keepdims=True) / n,
                           o0.sum(2, keepdims=True) / n, r1.sum(2, keepdims=True) / n,
                           o1.sum(2, keepdims=True) / n], axis=2)
    tgt0 = np.where(tr[safe] & obs0[safe] & pm, np.nan_to_num(y0[safe]) - m0, 0.0)
    tgt1 = np.where(tr[safe] & obs1[safe] & pm, np.nan_to_num(y1[safe]) - inc[safe], 0.0)
    w0 = (tr[safe] & obs0[safe] & pm).astype(np.float32)
    w1 = (tr[safe] & obs1[safe] & pm).astype(np.float32)
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    Xs_t, summ_t, t0, t1, w0t, w1t = f(Xs), f(summ), f(tgt0), f(tgt1), f(w0), f(w1)
    p0 = np.full(N, np.nan); p1 = np.full(N, np.nan)
    acc0 = np.zeros(N); acc1 = np.zeros(N)
    for seed in seeds:
        torch.manual_seed(seed)
        gru = torch.nn.GRU(summ.shape[2], hidden, batch_first=True)
        head = torch.nn.Sequential(torch.nn.Linear(hidden + D, 24), torch.nn.Tanh(), torch.nn.Linear(24, 2))
        opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=0.01, weight_decay=1e-3)

        def forward():
            h, _ = gru(summ_t)
            hp = torch.cat([torch.zeros(B, 1, hidden), h[:, :-1]], 1)
            z = torch.cat([hp[:, :, None, :].expand(B, T, Pm, hidden), Xs_t], 3)
            return head(z)

        for _ in range(epochs):
            opt.zero_grad()
            o = forward()
            loss = ((o[..., 0] - t0) ** 2 * w0t).sum() / w0t.sum().clamp(min=1) + \
                   ((o[..., 1] - t1) ** 2 * w1t).sum() / w1t.sum().clamp(min=1)
            loss.backward()
            opt.step()
        with torch.no_grad():
            o = forward().numpy()
        for b in range(B):
            for t in range(T):
                for p in range(Pm):
                    if idx[b, t, p] >= 0:
                        acc0[idx[b, t, p]] += o[b, t, p, 0] / len(seeds)
                        acc1[idx[b, t, p]] += o[b, t, p, 1] / len(seeds)
    p0 = np.clip(acc0 + m0, 0, 1)
    p1 = np.clip(acc1 + inc, 0, 1)
    return p0, p1


# --------------------------------------------------------------------------- metrics
def group_stats(y, pred, gid, n_groups):
    ok = ~np.isnan(y) & ~np.isnan(pred)
    s = {k: np.zeros(n_groups) for k in ("n", "abs", "err", "sq", "pc", "pn", "reg", "regn")}
    for g in range(n_groups):
        ix = np.where((gid == g) & ok)[0]
        if len(ix) == 0:
            continue
        e = pred[ix] - y[ix]
        s["n"][g] = len(ix); s["abs"][g] = np.abs(e).sum(); s["err"][g] = e.sum(); s["sq"][g] = (e ** 2).sum()
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                i, j = ix[a], ix[b]
                if y[i] == y[j]:
                    continue
                s["pn"][g] += 1
                sg = np.sign(pred[i] - pred[j])
                s["pc"][g] += float(sg == np.sign(y[i] - y[j])) + 0.5 * float(sg == 0)
        if len(ix) >= 2:
            top = ix[pred[ix] >= pred[ix].max() - 1e-12]
            s["reg"][g] = y[ix].max() - y[top].mean()
            s["regn"][g] = 1
    return s


def aggregate(s, w=None, calibrated=True):
    w = np.ones_like(s["n"]) if w is None else w
    n = float((w * s["n"]).sum()); pn = float((w * s["pn"]).sum()); rn = float((w * s["regn"]).sum())
    out = {"n": int(round(n)), "pair_count": int(round(pn)),
           "pair_accuracy": float((w * s["pc"]).sum() / pn) if pn else None,
           "top1_regret": float((w * s["reg"]).sum() / rn) if rn else None, "regret_rounds": int(round(rn))}
    if calibrated and n:
        out.update(mae=float((w * s["abs"]).sum() / n), bias=float((w * s["err"]).sum() / n),
                   rmse=float(np.sqrt((w * s["sq"]).sum() / n)))
    else:
        out.update(mae=None, bias=None, rmse=None)
    return out


def paired_bootstrap(sa, sb, calibrated, n_boot=1000, seed=0):
    G = len(sa["n"])
    live = np.where((sa["n"] > 0) | (sb["n"] > 0))[0]
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, G))
    for k in range(n_boot):
        W[k] = np.bincount(rng.choice(live, size=len(live)), minlength=G)
    res = {}
    for name in ("mae", "pair_accuracy", "top1_regret"):
        if name == "mae" and not calibrated:
            continue
        diffs = []
        for w in W:
            a, b = aggregate(sa, w, calibrated), aggregate(sb, w, calibrated)
            if a[name] is not None and b[name] is not None:
                diffs.append(a[name] - b[name])
        point = aggregate(sa, None, calibrated)[name], aggregate(sb, None, calibrated)[name]
        if diffs and None not in point:
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            res[name] = {"a": point[0], "b": point[1], "diff_a_minus_b": point[0] - point[1],
                         "ci95": [float(lo), float(hi)], "rounds": int(len(live))}
    return res


# --------------------------------------------------------------------------- holdouts
def make_schemes(records, min_group=3):
    skills = np.array([r["skill"] for r in records])
    N = len(records)
    fam = np.array([r["intervention_primary_field"] for r in records], dtype=object)
    counts = {f: int((fam == f).sum()) for f in set(fam)}
    fam = np.array([f if counts[f] >= min_group else "other" for f in fam], dtype=object)
    schemes = {}
    schemes["research_stopping"] = [("research+stopping", np.where(~np.isin(skills, PILOT_TEST_SKILLS))[0],
                                     np.where(np.isin(skills, PILOT_TEST_SKILLS))[0])]
    schemes["leave_skill_out"] = [(s, np.where(skills != s)[0], np.where(skills == s)[0])
                                  for s in SKILLS if (skills == s).any()]
    schemes["leave_intervention_out"] = [(f, np.where(fam != f)[0], np.where(fam == f)[0])
                                         for f in sorted(set(fam))]
    return schemes, fam


# --------------------------------------------------------------------------- experiment
CALIBRATED = {"train_mean": True, "incumbent": True, "incumbent_offset": True, "live_inky": True,
              "ridge_pilot": True, "ridge_full": True, "ridge_resid": True, "gru_controller": True,
              "gru_online": True, "pairwise_logit": False, "random_ranking": False,
              "heur_small_edit": False, "heur_promoted_fields": False, "heur_never_visits": False}
V3_MODELS = ("ridge_full", "ridge_resid", "gru_controller", "gru_online", "pairwise_logit")


def run_experiment(records, gru_epochs=100, n_boot=1000):
    N = len(records)
    feats = feature_matrices(records)
    gid_str = [r["group"] for r in records]
    gmap = {g: i for i, g in enumerate(dict.fromkeys(gid_str))}
    gid = np.array([gmap[g] for g in gid_str])
    G = len(gmap)
    inc = np.array([r["input"]["incumbent_dev_exact"] if r["input"]["incumbent_dev_exact"] is not None
                    else np.nan for r in records], float)
    inky = np.array([r["input"]["expected_effect"]["predicted_dev_exact"] if
                     r["input"]["expected_effect"]["predicted_dev_exact"] is not None else np.nan
                     for r in records], float)
    Y = {rg: np.array([np.nan if r["target"][rg + "_dev_exact"] is None else r["target"][rg + "_dev_exact"]
                       for r in records]) for rg in RUNGS}
    prof = [edit_profile(r["input"]["proposed_intervention"]["change"], r["input"]["current_architecture"])
            for r in records]
    edit = np.array([sum(p.values()) for p in prof])
    touches = np.array([float(any(k in SEARCH_FIELDS for k in p)) for p in prof])
    promoted = np.array([r["target"]["decision"] == "promoted_to_incumbent" for r in records])
    seqs = _sequences(records)
    schemes, fam = make_schemes(records)
    inc_fill = np.where(np.isnan(inc), np.nanmean(inc), inc)

    results = {"schemes": {}}
    oof_all = {}
    for sname, folds in schemes.items():
        oof = {rg: {} for rg in RUNGS}
        fold_rows = {rg: {} for rg in RUNGS}
        for fname, tr, te in folds:
            tr_mask = np.zeros(N, bool); tr_mask[tr] = True
            gru_cache = {}

            def gru_get(online):
                if online not in gru_cache:
                    visible = np.ones(N, bool) if online else tr_mask.copy()
                    gru_cache[online] = gru_predict(feats["compact"], Y["rung0"], Y["rung1"], inc_fill,
                                                    seqs, tr, visible, epochs=gru_epochs)
                return gru_cache[online]

            # promoted-rate per field from training rows only
            rate = {}
            for f in FIELDS:
                has = np.array([f in p for p in prof])
                m = tr_mask & has
                rate[f] = (promoted[m].sum() + 0.5) / (m.sum() + 2.0)
            heur_prom = np.array([np.mean([rate[f] for f in p]) if p else 0.0 for p in prof])
            pw = pairwise_fit(feats["compact"][tr][~np.isnan(Y["rung0"][tr])], Y["rung0"][tr][~np.isnan(Y["rung0"][tr])],
                              gid[tr][~np.isnan(Y["rung0"][tr])])
            for ri, rg in enumerate(RUNGS):
                y = Y[rg]
                a = tr[~np.isnan(y[tr])]
                preds = {}
                nan = np.full(N, np.nan)
                preds_all = {k: nan.copy() for k in CALIBRATED}
                if len(a) >= 5:
                    ym = float(y[a].mean())
                    preds_all["train_mean"][te] = ym
                    preds_all["incumbent"][te] = inc_fill[te]
                    off = float(np.mean(y[a] - inc_fill[a]))
                    preds_all["incumbent_offset"][te] = np.clip(inc_fill[te] + off, 0, 1)
                    preds_all["live_inky"][te] = inky[te]
                    preds_all["ridge_pilot"][te] = ridge_pilot_predict(feats["pilot"][a], y[a], feats["pilot"][te])
                    preds_all["ridge_full"][te] = ridge_cv_predict(feats["full"][a], y[a], feats["full"][te], gid[a])
                    preds_all["ridge_resid"][te] = np.clip(inc_fill[te] + ridge_cv_predict(
                        feats["compact"][a], y[a] - inc_fill[a] + 0.5, feats["compact"][te], gid[a]) - 0.5, 0, 1)
                    g0, g1 = gru_get(False)
                    preds_all["gru_controller"][te] = (g0, g1)[ri][te]
                    if sname != "leave_intervention_out":
                        o0, o1 = gru_get(True)
                        preds_all["gru_online"][te] = (o0, o1)[ri][te]
                    preds_all["pairwise_logit"][te] = pairwise_score(pw, feats["compact"][te])
                    preds_all["random_ranking"][te] = 0.0
                    preds_all["heur_small_edit"][te] = -edit[te]
                    preds_all["heur_promoted_fields"][te] = heur_prom[te]
                    preds_all["heur_never_visits"][te] = -touches[te]
                for k, v in preds_all.items():
                    oof[rg].setdefault(k, np.full(N, np.nan))
                    oof[rg][k][te] = v[te]
                fold_rows[rg][fname] = te
        oof_all[sname] = oof
        block = {"folds": [f[0] for f in folds], "test_rows": int(sum(len(f[2]) for f in folds)), "rungs": {}}
        for rg in RUNGS:
            y = Y[rg]
            mt = {}
            for k, p in oof[rg].items():
                s = group_stats(y, p, gid, G)
                m = aggregate(s, None, CALIBRATED[k])
                m["per_fold_mae_pp"] = {fn: (None if not CALIBRATED[k] else _fold_mae(y, p, ix))
                                        for fn, ix in fold_rows[rg].items()}
                mt[k] = _to_pp(m)
            block["rungs"][rg] = {"observed_test_rows": int((~np.isnan(y[np.concatenate([f[2] for f in folds])])).sum()),
                                  "models": mt}
        results["schemes"][sname] = block

    # model selection (descriptive, uses test numbers: flagged in caveats)
    best = {}
    for rg in RUNGS:
        mae = {m: np.mean([results["schemes"][s]["rungs"][rg]["models"][m]["mae_pp"]
                           for s in results["schemes"]]) for m in V3_MODELS if CALIBRATED[m]
               and all(results["schemes"][s]["rungs"][rg]["models"][m]["mae_pp"] is not None
                       for s in results["schemes"])}
        acc = {}
        for m in V3_MODELS:
            vals = [results["schemes"][s]["rungs"][rg]["models"][m]["pair_accuracy"] for s in results["schemes"]]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(results["schemes"]) or (vals and m != "gru_online"):
                acc[m] = float(np.mean(vals))
        best[rg] = {"best_mae_model": min(mae, key=mae.get) if mae else None,
                    "mean_mae_pp": mae, "best_rank_model": max(acc, key=acc.get) if acc else None, "mean_pair_accuracy": acc}
    results["selection"] = best

    # paired bootstrap over rounds
    boot = {}
    for sname, oof in oof_all.items():
        for rg in RUNGS:
            y = Y[rg]
            sel = {best[rg]["best_mae_model"], best[rg]["best_rank_model"]} - {None}
            for m in sorted(sel):
                for base in ("ridge_pilot", "incumbent", "live_inky"):
                    pa, pb = oof[rg][m].copy(), oof[rg][base].copy()
                    bad = np.isnan(pa) | np.isnan(pb)
                    pa[bad] = np.nan; pb[bad] = np.nan
                    cal = CALIBRATED[m] and CALIBRATED[base]
                    r = paired_bootstrap(group_stats(y, pa, gid, G), group_stats(y, pb, gid, G), cal, n_boot)
                    for key in ("mae", "top1_regret"):
                        if key in r:
                            v = r[key]
                            r[key] = {"a_pp": v["a"] * 100, "b_pp": v["b"] * 100,
                                      "diff_a_minus_b_pp": v["diff_a_minus_b"] * 100,
                                      "ci95_pp": [c * 100 for c in v["ci95"]], "rounds": v["rounds"]}
                    boot[f"{sname}/{rg}/{m}_vs_{base}"] = r
    results["paired_bootstrap"] = boot
    return results, oof_all, schemes, fam, gid, G


def _fold_mae(y, p, ix):
    ok = ~np.isnan(y[ix]) & ~np.isnan(p[ix])
    return float(np.abs(y[ix][ok] - p[ix][ok]).mean() * 100) if ok.any() else None


def _to_pp(m):
    for k in ("mae", "bias", "rmse"):
        m[k + "_pp"] = None if m[k] is None else m[k] * 100
        del m[k]
    if m["top1_regret"] is not None:
        m["top1_regret_pp"] = m.pop("top1_regret") * 100
    else:
        m["top1_regret_pp"] = m.pop("top1_regret")
    return m


def pilot_reproduction(records):
    """Pilot ridge (alpha 10, features and holdout as in trm_skill_gym_v2) via this module's code."""
    P = np.stack([_pilot_features(r["input"]) for r in records])
    skills = np.array([r["skill"] for r in records])
    te_m = np.isin(skills, PILOT_TEST_SKILLS)
    gid = {g: i for i, g in enumerate(dict.fromkeys(r["group"] for r in records))}
    gids = np.array([gid[r["group"]] for r in records])
    out = {}
    for rg in RUNGS:
        y = np.array([np.nan if r["target"][rg + "_dev_exact"] is None else r["target"][rg + "_dev_exact"]
                      for r in records])
        a = np.where(~te_m & ~np.isnan(y))[0]; b = np.where(te_m & ~np.isnan(y))[0]
        p = np.full(len(y), np.nan)
        p[b] = ridge_pilot_predict(P[a], y[a], P[b])
        m = aggregate(group_stats(y, p, gids, len(gid)))
        out[rg] = {"n_test": m["n"], "mae_pp": m["mae"] * 100, "bias_pp": m["bias"] * 100,
                   "rmse_pp": m["rmse"] * 100, "pair_accuracy": m["pair_accuracy"], "pair_count": m["pair_count"]}
    return out


# --------------------------------------------------------------------------- export
def export_records(records, schemes, fam):
    split = [dict() for _ in records]
    for sname, folds in schemes.items():
        for fname, tr, te in folds:
            for i in te:
                split[i][sname] = fname
    pilot_te = {i for f in schemes["research_stopping"] for i in f[2]}
    out = []
    for i, r in enumerate(records):
        m = r["export_meta"]
        cfg = r["input"]["proposed_intervention"]["config"] or {}
        res = r["input"]["predicted_resource_use"]
        sp = dict(split[i])
        sp["research_stopping"] = "test" if i in pilot_te else "train"
        out.append({
            "schema_version": "1.0", "record_schema": "goblinbench.trm_skill_gym.distill.v3",
            "candidate_id": m["candidate_id"], "architecture_family": m["architecture_family"],
            "mock": False, "weights_loaded": False, "training_splits": ["train"],
            "training_split_hashes": m["training_split_hashes"],
            "dataset_manifest_sha256": m["dataset_manifest_sha256"],
            "manifest_source": m["manifest_source"],
            "maximum_loops": int(cfg.get("visits") or r["input"]["current_architecture"].get("visits") or 1),
            "resource_summary": {"n_params": res["n_params"], "flops_proxy": res["flops_proxy"]},
            "id": r["id"], "skill": r["skill"], "arm": r["arm"],
            "intervention_primary_field": r["intervention_primary_field"],
            "input": r["input"], "target": r["target"], "split": sp})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--extra-archive", nargs="*", default=[])
    ap.add_argument("--gru-epochs", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(argv)
    archives = [("gym01", a.archive)] + [(f"extra{i}", p) for i, p in enumerate(a.extra_archive)]
    records = build_records(archives)
    results, _, schemes, fam, _, _ = run_experiment(records, a.gru_epochs, a.n_boot)
    results["n_records"] = len(records)
    results["sources"] = [{"label": l, "path": str(p), "sha256_lf": file_sha256(p)} for l, p in archives]
    results["pilot_reproduction"] = pilot_reproduction(records)
    results["caveats"] = [
        "108 proposals (36 rounds); rung1 exists for 36 rows; all intervals are wide.",
        "Best-model selection and the model comparison use the same out-of-fold test numbers; treat as descriptive.",
        "random_ranking is the exact expectation of a seeded random ranking (constant scores, ties scored 0.5, tie-averaged regret).",
        "Intervention-family pairs pool predictions from different fold models; within-round pairs across folds are not strictly like-for-like.",
        "Heuristic and pairwise scores are rank-only; no MAE is reported.",
        "Rung1 ranking pairs are few; pairwise ranker is trained on rung0 pairs only.",
        "live_inky exists only for the two Inky arms; bootstrap comparisons against it use that subset for both models.",
        "Features that were imputed (missing Inky forecast, missing params) carry indicator flags; targets are never imputed."]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    with open(out / "distill_records.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in export_records(records, schemes, fam):
            f.write(json.dumps(r, sort_keys=True) + "\n")
    for s, b in results["schemes"].items():
        for rg in RUNGS:
            line = {m: (None if v["mae_pp"] is None else round(v["mae_pp"], 2), None if v["pair_accuracy"] is None else round(v["pair_accuracy"], 3))
                    for m, v in b["rungs"][rg]["models"].items()}
            print(s, rg, line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
