import json
import sys

import numpy as np
import pytest
import torch

from research_gym.trm_skill_gym import tasks as T
from research_gym.trm_skill_gym_v3 import graphs as G
from research_gym.trm_skill_gym_v3 import hybrid as H
from research_gym.trm_skill_gym_v3 import supervisor as SUP


@pytest.mark.parametrize("skill", G.GRAPH_SKILLS)
def test_graph_strata_labels_verify_and_are_deterministic(skill):
    for fam in G.strata(skill, "test"):
        b = T.stack([G.instance(skill, fam, 7_000_000 + i) for i in range(12)])
        ok, _ = T.verify(skill, b, b["label"])
        assert ok.all(), fam
        b2 = T.stack([G.instance(skill, fam, 7_000_000 + i) for i in range(12)])
        assert torch.equal(b["rel"], b2["rel"]) and torch.equal(b["label"], b2["label"])
        assert T.certificate_violations(skill, b) == 0


@pytest.mark.parametrize("skill", G.GRAPH_SKILLS)
def test_hintfree_neutralises_label_correlated_columns(skill):
    fam = T.FAMILIES[skill]["dev"][0]
    b = T.stack([G.instance(skill, fam + "_hintfree", 7_100_000 + i) for i in range(32)])
    for c in G.HINT_COLS[skill]:
        vals = b["cand"][..., c][b["mask"] > 0]
        assert torch.allclose(vals, torch.full_like(vals, 0.5))


def test_topology_permutes_slots():
    a = G.instance("retrieval", "long_chain_10", 7_200_000)
    b = G.instance("retrieval", "long_chain_10", 7_200_001)
    assert int(np.argmax(a.cand[:, 0])) != int(np.argmax(b.cand[:, 0])) or not np.array_equal(a.rel, b.rel)


def test_withheld_topologies_not_in_training_or_calibration():
    for skill in G.GRAPH_SKILLS:
        train = set(G.strata(skill, "train"))
        assert train == set(G.strata(skill, "cal"))
        assert not train & set(G.WITHHELD_TOPOLOGIES)
        assert not train & set(T.FAMILIES[skill]["withheld"])


@pytest.mark.parametrize("skill", G.GRAPH_SKILLS)
def test_three_valued_certificates_are_sound_over_all_realizations(skill):
    checked = 0
    for s in H.strata(skill, "test"):
        if s.startswith("complete"):
            continue
        for i in range(3):
            ob = H.soundness_obligation(skill, H.make(skill, s, 7_300_000 + i))
            if ob["checked"]:
                checked += 1
                assert ob["violations"] == 0, s
    assert checked >= 10


@pytest.mark.parametrize("skill", G.GRAPH_SKILLS)
def test_complete_observations_are_fully_certified_and_exact(skill):
    b = T.stack([H.make(skill, "complete:branching", 7_400_000 + i) for i in range(16)])
    _, cert, work = H.annotate(skill, b)
    assert bool(((cert != 0) | (b["mask"] == 0)).all()) and work > 0
    assert T.exact(skill, b, (cert > 0).float()).mean() == 1.0


def test_certificate_inputs_exclude_labels():
    inst = H.make("deduction", "noisy:kernel_h4", 7_500_000)
    st1, _, _ = H.certify3("deduction", inst.rel, inst.mask, None)
    inst.label[:] = 1 - inst.label
    inst.aux[:] = 0
    st2, _, _ = H.certify3("deduction", inst.rel, inst.mask, None)
    assert np.array_equal(st1, st2)


def test_supervisor_kills_process_tree_over_memory_limit(tmp_path):
    code = "import time\nx = bytearray(400 * 1024 * 1024)\ntime.sleep(30)\n"
    res = SUP.run([sys.executable, "-c", code], wall_s=60, limit_bytes=150 * 1024 * 1024, period=0.2)
    assert res["status"] == "memory_killed"
    res = SUP.run([sys.executable, "-c", "print(1)"], wall_s=60, limit_bytes=2 * 1024**3)
    assert res["status"] == "ok"
    res = SUP.run([sys.executable, "-c", "import time; time.sleep(20)"], wall_s=2, limit_bytes=2 * 1024**3)
    assert res["status"] == "wall_killed"


def test_race_roster_includes_matched_damping_and_rlt():
    from research_gym.trm_skill_gym_v3 import race as R

    archive = [json.loads(l) for l in (R.SOURCE / "search" / "archive.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ro = R.roster(archive, "repair", 2)
    for k in R.FIXED:
        assert k in ro
    assert sum(k.startswith("archive_") for k in ro) <= 2
