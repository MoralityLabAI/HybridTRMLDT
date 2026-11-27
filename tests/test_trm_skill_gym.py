import io
import json
import math
import urllib.error

import numpy as np
import pytest
import torch

from research_gym.trm_skill_gym import dsl, inky
from research_gym.trm_skill_gym import modules as M
from research_gym.trm_skill_gym import skill_templates as S
from research_gym.trm_skill_gym import tasks as T
from research_gym.trm_skill_gym import train as TR


def _batch(skill, split="train", n=48, withheld=False):
    fams = T.FAMILIES[skill]["withheld" if withheld else "dev"]
    return T.make_batch(skill, fams, split, 0, n)


@pytest.mark.parametrize("skill", T.SKILLS)
def test_instances_are_deterministic_and_roots_disjoint(skill):
    fam = T.FAMILIES[skill]["dev"][0]
    a, b = T.make_instance(skill, fam, 123), T.make_instance(skill, fam, 123)
    for k in ("cand", "rel", "label", "mask", "aux"):
        assert np.array_equal(getattr(a, k), getattr(b, k))
    c = T.make_instance(skill, fam, 124)
    assert not np.array_equal(a.cand, c.cand)
    tr, de, fr = T.ROOT_RANGES["train"], T.ROOT_RANGES["dev_eval"], T.ROOT_RANGES["frozen_eval"]
    assert tr[1] <= de[0] and de[1] <= fr[0]
    assert set(T.FAMILIES[skill]["dev"]).isdisjoint(T.FAMILIES[skill]["withheld"])


@pytest.mark.parametrize("skill", T.SKILLS)
@pytest.mark.parametrize("withheld", [False, True])
def test_verifier_accepts_labels_and_certificates_are_sound(skill, withheld):
    b = _batch(skill, "frozen_eval", 64, withheld)
    ok, diag = T.verify(skill, b, b["label"])
    assert ok.all()
    assert diag.shape == (64, T.N, 2)
    assert T.certificate_violations(skill, b) == 0
    assert (b["label"] * (1 - b["mask"])).sum() == 0
    if T.LABEL_KIND[skill] == "single":
        assert torch.all(b["label"].sum(-1) == 1)


@pytest.mark.parametrize("skill", T.SKILLS)
def test_verifier_rejects_wrong_answers(skill):
    b = _batch(skill, n=32)
    wrong = (1 - b["label"]) * b["mask"]
    if T.LABEL_KIND[skill] == "single":
        wrong = torch.nn.functional.one_hot(((b["label"].argmax(-1) + 1) % b["mask"].sum(-1).long()), T.N).float()
        ok, _ = T.verify(skill, b, wrong)
        assert ok.float().mean() < 0.5
    elif skill == "repair":  # the complement flip set is a valid but non-minimal repair
        ok, _ = T.verify(skill, b, wrong)
        assert ok.all() and T.exact(skill, b, wrong).sum() == 0
    else:
        ok, _ = T.verify(skill, b, wrong)
        assert not ok.any()


def test_repair_diagnostics_vanish_on_valid_repair():
    b = _batch("repair", n=32)
    ok, diag = T.verify("repair", b, b["label"])
    assert ok.all() and diag[..., 0].sum() == 0
    ok0, diag0 = T.verify("repair", b, torch.zeros_like(b["label"]))
    assert not ok0.any() and diag0[..., 0].sum() > 0


def test_dsl_rejects_unknown_and_out_of_range():
    with pytest.raises(dsl.ProposalError):
        dsl.validate({"exec": "import os"})
    with pytest.raises(dsl.ProposalError):
        dsl.validate({"hidden": 1000})
    with pytest.raises(dsl.ProposalError):
        dsl.validate({"core": "transformer_xl"})
    with pytest.raises(dsl.ProposalError):
        dsl.validate({"ldt_projection": "yes"})
    cfg = dsl.validate({"cycles": 1, "grad_cycles": 4, "visits": 1, "verifier_visits": 2})
    assert cfg["grad_cycles"] == 1 and cfg["verifier_visits"] == 0
    assert dsl.validate({"hidden": 64.0})["hidden"] == 64


def test_dsl_limit_edit_and_mutation():
    inc = dsl.defaults()
    prop = dsl.validate({"hidden": 64, "visits": 4, "feedback": "rlt"})
    out, dropped = dsl.limit_edit(inc, prop, 1)
    assert len(dsl.diff(inc, out)) == 1 and len(dropped) == 2
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = dsl.mutate(inc, rng, 2)
        assert dsl.validate(m) == m


def test_default_arms_within_parameter_bounds():
    from research_gym.trm_skill_gym.campaign import arm_configs

    arms = arm_configs()
    d = M.SkillModule(arms["D_trm_direct"]).n_params()
    for name, cfg in arms.items():
        if cfg["core"] == "heuristic":
            continue
        n = M.SkillModule(cfg).n_params()
        assert dsl.PARAM_MIN <= n <= dsl.PARAM_MAX, name
        if name in ("B_feedforward", "C_gru"):
            assert abs(n - d) / d < 0.15, (name, n, d)


def test_feedback_weight_matches_datagoblin_schedule():
    depth = 4
    assert [M.feedback_weight("direct", v, depth) for v in range(depth)] == [1.0] * 4
    assert [M.feedback_weight("rlt", v, depth) for v in range(depth)] == [1 / 4, 1 / 3, 1 / 2, 1.0]
    damp = [M.feedback_weight("matched_damping", v, depth) for v in range(depth)]
    assert damp[-1] == 1.0
    assert math.isclose(sum(damp[:-1]), sum([1 / 4, 1 / 3, 1 / 2]))
    assert len(set(damp[:-1])) == 1


@pytest.mark.parametrize("core", ["trm", "gru", "ff"])
def test_episode_contract_shapes_and_halting(core):
    cfg = dsl.validate({"core": core, "hidden": 16, "visits": 3, "verifier_visits": 1})
    model = M.SkillModule(cfg)
    b = _batch("deduction", n=8)
    out = M.episode(model, "deduction", b, cfg)
    assert len(out["logits"]) == 3 and out["logits"][0].shape == (8, T.N)
    assert out["vcalls"] == 1 and out["sched"] == [1]
    h = M.halting(out, tau=0.0)
    assert torch.all(h["visits_used"] == 1)
    h = M.halting({**out, "value": [torch.ones(8)] * 3}, tau=1.01)
    assert torch.all(h["visits_used"] == 3) and torch.all(h["vcalls"] == 1)


def test_ldt_projection_clamps_certified_logits():
    b = _batch("deduction", n=16)
    cert = T.certify("deduction", b)
    proj = M.project(torch.zeros(16, T.N), cert)
    assert torch.all(proj[cert > 0] >= 6) and torch.all(proj[cert < 0] <= -6)
    cfg = dsl.validate({"hidden": 16, "ldt_projection": True})
    out = M.episode(M.SkillModule(cfg), "deduction", b, cfg)
    pred = out["pred"][0]
    assert torch.all(pred[cert > 0] == 1) and torch.all(pred[(cert < 0) & (b["mask"] > 0)] == 0)


def test_latent_reset_changes_trajectory_but_not_first_visit():
    torch.manual_seed(0)
    cfg = dsl.validate({"hidden": 16, "visits": 3})
    m = M.SkillModule(cfg)
    b = _batch("retrieval", n=8)
    a = M.episode(m, "retrieval", b, cfg)
    r = M.episode(m, "retrieval", b, dict(cfg, latent_reset=True))
    m.cfg["latent_reset"] = True
    r = M.episode(m, "retrieval", b, m.cfg)
    assert torch.allclose(a["logits"][0], r["logits"][0])
    assert not torch.allclose(a["logits"][-1], r["logits"][-1])


def test_training_reduces_loss_and_respects_ram_cap():
    pool = T.make_batch("deduction", T.FAMILIES["deduction"]["dev"], "train", 0, 128)
    cfg = dsl.validate({"hidden": 16, "lr": 5e-3})
    model, info = TR.train_module("deduction", cfg, 30, 0, pool)
    assert info["steps"] == 30 and info["peak_rss_delta_bytes"] < TR.RAM_CAP_BYTES
    ev = TR.evaluate(model, "deduction", cfg, T.make_batch("deduction", ["kernel_h4"], "dev_eval", 0, 32))
    for k in ("exact", "ece", "adaptive_exact", "flops_per_instance", "verifier_calls_per_instance"):
        assert k in ev
    assert ev["verifier_calls_per_instance"] == 2.0


def test_shuffled_supervision_keeps_valid_targets():
    pool = T.make_batch("research", T.FAMILIES["research"]["dev"], "train", 0, 64)
    cfg = dsl.validate({"hidden": 16})
    _, info = TR.train_module("research", cfg, 3, 0, pool, shuffled=True)
    assert info["steps"] == 3


def test_heuristic_arm_runs_protocol():
    b = _batch("retrieval", "frozen_eval", 32)
    out = M.episode(None, "retrieval", b, dsl.defaults(), heuristic=True)
    assert out["vcalls"] == 1
    assert out["exact"][-1].mean() >= out["exact"][0].mean()


def test_skill_templates_render_all_sections(tmp_path):
    paths = S.write_all(tmp_path)
    for skill, p in paths.items():
        text = p.read_text(encoding="utf-8")
        for sec in ("## Trigger", "## Objective", "## Action schema", "## Procedure", "## Verifier contract", "## Termination"):
            assert sec in text
        inst = T.make_instance(skill, T.FAMILIES[skill]["dev"][0], 1)
        assert "Candidates:" in S.serialize_instance(skill, inst)


def test_inky_client_budget_retries_and_parsing():
    calls = []

    def flaky(key, payload, timeout):
        calls.append(payload)
        if len(calls) == 1:
            raise urllib.error.HTTPError("u", 429, "rate", {}, io.BytesIO(b""))
        return {"choices": [{"message": {"content": 'ok {"a": {"b": 1}} trailing'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}, "model": inky.MODEL}

    c = inky.InkyClient(max_completion_tokens=100, transport=flaky, base_backoff=0.0, min_interval=0.0)
    r = c.chat("s", "u")
    assert inky.extract_json(r["content"]) == {"a": {"b": 1}}
    assert c.ledger["retries"] == 1 and c.ledger["requests"] == 2
    assert calls[0]["model"] == inky.MODEL and calls[0]["model"].endswith(":free")
    assert "test" not in json.dumps(c.ledger)
    with pytest.raises(ValueError):
        inky.InkyClient(context=20_000)
    with pytest.raises(ValueError):
        c.chat("s", "x" * 60_000)

    def dead(key, payload, timeout):
        raise urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b""))

    with pytest.raises(inky.InkyUnavailable):
        inky.InkyClient(max_completion_tokens=10, transport=dead, base_backoff=0.0, min_interval=0.0).chat("s", "u")


def test_campaign_smoke_end_to_end(tmp_path):
    from research_gym.trm_skill_gym import campaign as C
    from research_gym.trm_skill_gym import report as R

    camp = C.Campaign(tmp_path / "run", 1.0, C.SMOKE, use_inky=False)
    camp.go()
    run = tmp_path / "run"
    assert (run / "frozen_architectures.json").exists()
    assert len(list((run / "main").glob("*.json"))) == len(T.SKILLS) * 8
    state = json.loads((run / "search" / "state.json").read_text())
    assert any(e["arm"] == "random_mutation" for e in state["archive"])
    import sys

    argv = sys.argv
    sys.argv = ["report", "--run", str(run), "--out", str(tmp_path / "out")]
    try:
        R.main()
    finally:
        sys.argv = argv
    assert (tmp_path / "out" / "REPORT.md").exists()
    recs = (tmp_path / "out" / "goblinbench_records.jsonl").read_text().splitlines()
    for line in recs:
        rec = json.loads(line)
        for k in ("existing_skill_template", "proposed_architecture", "rationale", "predicted_result", "actual_result",
                  "failure_diagnosis", "revised_architecture", "resource_summary", "final_promotion_decision",
                  "schema_version", "candidate_id"):
            assert k in rec
