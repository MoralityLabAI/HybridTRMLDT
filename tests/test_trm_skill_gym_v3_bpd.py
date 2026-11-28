"""BPD of TRM skill modules: exactness, mean-write ablation, determinism, receipt replay (CPU, < 2 min)."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research_gym.trm_skill_gym import modules as M, tasks as T
from research_gym.trm_skill_gym_v3 import bpd_skill as B

CKPT = Path("experiments/trm_skill_gym_01/run/checkpoints/repair__F_trm_rlt__s0.pt")

pytestmark = [
    pytest.mark.skipif(B.BPD is None, reason=f"BPD library not importable: {B.BPD_ERROR}"),
    pytest.mark.skipif(not CKPT.is_file(), reason="GYM-01 checkpoint missing"),
]


@pytest.fixture(scope="module")
def rig():
    torch.set_num_threads(1)
    hp = B.HP.smoke()
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    r = B.Rig("repair", ck, hp)
    r.pool = B.pools("repair", B.FIT, 8)
    r.calibrate("bpd", r.pool, [r.R, 2 * r.R])
    r.calibrate("svd", r.pool, [r.R, 2 * r.R])
    return r


def test_bpd_commit_recorded():
    sha = B.bpd_commit()
    assert sha and len(sha) == 40


def test_target_mode_reproduces_original_outputs(rig):
    batch = B.cat_pool(rig.pool)
    with torch.no_grad():
        ref = M.episode(rig.plain, "repair", batch, rig.cfg)
        for kind in ("bpd", "svd"):
            model, bm = rig.parts(kind)
            bm.clear_schedules()
            bm.reset_wrappers()
            out = M.episode(model, "repair", batch, rig.cfg)
            for a, b in zip(ref["logits"], out["logits"]):
                assert float((a - b).abs().max()) < 1e-5
            assert [float(x.sum()) for x in ref["exact"]] == [float(x.sum()) for x in out["exact"]]


def test_residual_is_exact_and_component_mode_with_open_masks_matches_target(rig):
    comp = B.require_bpd().components
    for kind in ("bpd", "svd"):
        _, bm = rig.parts(kind)
        for name, w in bm.wrappers.items():
            recon = (w.reconstructed_weight() + w.weight_delta()).detach()
            assert float((recon - w.W_target.float()).abs().max()) < 1e-5
            x = torch.randn(3, T.N, w.d_in)
            sched = comp.PassMaskSchedule(comp_mask=torch.ones(3, 1, T.N, w.C), delta_mask=torch.ones(3, 1, T.N))
            w.reset_pass_state()
            w.mode, w.schedule = "component", sched
            try:
                got = w(x)
            finally:
                w.mode, w.schedule = "target", None
            want = torch.nn.functional.linear(x, w.W_target, w.bias)
            assert float((got - want).abs().max()) < 1e-4
    assert sum(w.C for w in rig.sm.wrappers.values()) == sum(min(w.W_target.shape) for w in rig.sm.wrappers.values())


def test_mean_write_ablation_shapes_and_pass_selectivity(rig):
    comp = B.require_bpd().components
    n_passes = rig.calls_per_visit() * rig.R
    rig.use_means("bpd", rig.R)
    name = rig.bm.module_order[0]
    w = rig.bm.wrappers[name]
    assert tuple(w.mean_comp_acts[:n_passes].shape) == (n_passes, w.C)
    assert tuple(w.mean_delta_out[:n_passes].shape) == (n_passes, w.d_out)
    assert bool(w.calibrated_passes[:n_passes].all())
    x = torch.randn(2, T.N, w.d_in)
    comps = (0, 3)
    w.reset_pass_state()
    w.intervention = comp.ComponentIntervention(components=comps, passes=frozenset({1}), mode="ablate", scale=0.0)
    try:
        outs = [w(x) for _ in range(3)]
    finally:
        w.intervention = None
    target = torch.nn.functional.linear(x, w.W_target, w.bias)
    assert float((outs[0] - target).abs().max()) == 0.0 and float((outs[2] - target).abs().max()) == 0.0
    idx = list(comps)
    expect = (w.mean_comp_acts[1, idx] - x @ w.V[:, idx]) @ w.U[idx]
    assert float((outs[1] - target - expect).abs().max()) < 1e-4
    assert outs[1].shape == target.shape and float((outs[1] - target).abs().max()) > 0


def test_intervention_evaluation_is_deterministic(rig):
    name = rig.sm.module_order[0]
    spec = {name: [0, 1, 2]}
    a1, e1 = rig.eval_weights("svd", rig.pool, spec)
    a2, e2 = rig.eval_weights("svd", rig.pool, spec)
    assert B.acc_digest(B.flat(a1)) == B.acc_digest(B.flat(a2)) and e1 == e2 and e1 > 0
    base, e0 = rig.eval_weights("svd", rig.pool, {})
    plain, _ = rig.eval_latent(rig.pool, None)
    assert e0 == 0.0 and B.acc_digest(B.flat(base)) == B.acc_digest(B.flat(plain))
    # deeper rollouts use their own pass-conditional means and the same factors
    deep, _ = rig.eval_weights("svd", rig.pool, spec, visits=2 * rig.R)
    assert set(deep) == set(a1)


def test_cli_smoke_resume_and_receipt_replay(tmp_path):
    out = tmp_path / "run"
    args = ["--out", str(out), "--skills", "repair", "--seeds", "0", "--smoke"]
    res = B.main(args)
    assert res["complete"] and not res["incomplete_jobs"]
    assert res["bpd_commit"] == B.bpd_commit()
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    job = results["jobs"]["repair__F_trm_rlt__s0"]
    for m in ("A", "B", "C", "D", "E", "F", "H", "G_rank_0", "G_energy_0", "F@2x", "F@4x"):
        assert m in job["methods"]
        assert "root_bootstrap_95" in job["methods"][m]["paired_vs_A"]["pooled"]
    assert job["target_mode_max_logit_error"]["bpd"] < 1e-5 and len(job["per_pass"]["single_pass_deltas"]) == job["n_passes"]
    jd = out / "jobs" / "repair__F_trm_rlt__s0"
    for f in ("factors.pt", "importance.json", "result.json", "receipt.json"):
        assert (jd / f).is_file()
    assert b"\r" not in (jd / "receipt.json").read_bytes()
    rc = json.loads((jd / "receipt.json").read_text(encoding="utf-8"))
    assert rc["inputs"]["checkpoint_sha256"] and rc["inputs"]["bpd_commit"] == B.bpd_commit()
    assert rc["inputs"]["roots"]["test"][0] == 2_500_000
    done = B.replay(jd / "receipt.json", sample=3)
    assert "A" in done["replayed"] and "F" in done["replayed"]
    # resume: a finished job is not recomputed
    before = (jd / "result.json").read_bytes()
    B.main(args)
    assert (jd / "result.json").read_bytes() == before
    # a tampered expectation must fail replay
    bad = copy.deepcopy(rc)
    bad["replay_samples"][0]["exact"] = {k: v + 0.5 for k, v in bad["replay_samples"][0]["exact"].items()}
    (jd / "receipt.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(AssertionError):
        B.replay(jd / "receipt.json", sample=1)
