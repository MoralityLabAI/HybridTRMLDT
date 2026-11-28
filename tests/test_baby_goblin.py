import json

import numpy as np
import pytest
import torch

from research_gym.baby_goblin import build_registry as BR
from research_gym.baby_goblin.core import Registry, SkillSpec, Trajectory, datagoblin_events, pocket_goblin_call, pocket_goblin_tools, route
from research_gym.trm_skill_gym import tasks as T
from research_gym.trm_skill_gym_v3 import hybrid as H
from research_gym.trm_skill_gym_v3 import router as R


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    p = tmp_path_factory.mktemp("reg") / "registry.json"
    p.write_text(json.dumps(BR.build(tmp_path_factory.mktemp("empty"))), encoding="utf-8")
    return Registry.load(p)


@pytest.mark.parametrize("skill,family", [("repair", "tree_region"), ("repair", "cycle_region"), ("stopping", "horizon2"),
                                          ("stopping", "horizon4"), ("research", "uniform_prior"), ("research", "peaked_prior")])
def test_symbolic_solvers_are_exact_on_complete_observations(skill, family):
    for i in range(40):
        inst = T.make_instance(skill, family, 8_000_000 + i)
        r = R.run_expert(skill, "symbolic", inst, {})
        assert r["correct"] == 1.0, (skill, family, i)


@pytest.mark.parametrize("skill", ["retrieval", "deduction"])
def test_graph_solver_exact_on_complete_graphs(skill):
    for s in [x for x in H.strata(skill, "test") if x.startswith("complete")]:
        for i in range(6):
            inst = H.make(skill, s, 8_100_000 + i)
            assert R.run_expert(skill, "symbolic", inst, {})["correct"] == 1.0, s


def test_learned_skills_start_experimental_without_campaign_evidence(registry):
    for s in registry.specs.values():
        if s.kind in ("learned", "llm_fallback"):
            assert s.status == "experimental"


def test_route_never_stops_on_partially_resolved_certificate(registry):
    traj = Trajectory()
    inst = H.make("retrieval", "noisy:branching", 8_200_000)
    route(registry, "retrieval", inst, 10_000_000, traj)
    stops = [r for r in traj.rows if r.get("event") == "terminate_confident"]
    for st in stops:
        ex = [r for r in traj.rows if r.get("expert") == st["expert"] and "uncertainty" in r][-1]
        if ex["kind"] in ("certificate", "symbolic"):
            assert ex["uncertainty"] == 0.0


def test_hooks_match_external_formats(registry):
    traj = Trajectory()
    route(registry, "repair", T.make_instance("repair", "tree_region", 8_300_000), 10_000_000, traj)
    ev = datagoblin_events(traj)
    assert ev[0]["event"] == "start" and ev[-1]["event"] == "complete"
    assert all({"ts", "event"} <= set(e) and e["event"] in {"start", "checkpoint", "complete"} for e in ev)
    tools = pocket_goblin_tools(registry)
    assert all(set(t) == {"name", "description", "inputSchema"} for t in tools)
    inst = T.make_instance("repair", "tree_region", 8_300_001)
    out = pocket_goblin_call(registry, "skill_repair_symbolic", {"cand": inst.cand.tolist(), "rel": inst.rel.tolist(),
                                                                 "glob": inst.glob.tolist(), "mask": inst.mask.tolist()})
    dec = np.zeros(T.N, np.float32)
    dec[out["selected"]] = 1
    assert T.exact("repair", T.stack([inst]), torch.from_numpy(dec)[None]).item() == 1.0


def test_skills_are_independently_replaceable(registry):
    specs = list(registry.specs.values())
    swapped = Registry([s if s.name != "repair_learned" else SkillSpec(**{**s.__dict__, "kind": "heuristic", "implementation": "swap"}) for s in specs])
    traj = Trajectory()
    res = route(swapped, "repair", T.make_instance("repair", "tree_region", 8_400_000), 10_000_000, traj)
    assert res is not None
