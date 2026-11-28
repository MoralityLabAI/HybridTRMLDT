import json
import time

from . import analysis3, campaign3, export3, finals3, train3
from .factory3 import Config
from .gym3 import TaskSpec
from .test_campaign3 import _tiny


def test_freeze_finals_analysis_export(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run = tmp_path / "run"
    campaign3.Campaign(run, hours=1, threads=1, use_inky=False, max_iterations=2).run()
    base = Config(width=16, loops=2, steps=16)
    hrm = base.replace(routing="hrm", n_types=4)
    arms = {"A_single": base, "B_hrm": hrm, "H_sparse_st": hrm.replace(routing="sparse_st"),
            "K_distill": base.replace(distill="logits_all", distill_coef=1.0), "A_single_last": base.replace(loss="last")}
    spec = TaskSpec("proof_resolve", 6, depth=2)
    panel = {"train": spec, "eval": {"dev": spec, "ood_deep": TaskSpec("proof_resolve", 8, depth=3)}}
    monkeypatch.setattr(finals3, "resolved_arms", lambda: dict(arms))
    monkeypatch.setattr(finals3, "ARMS", {"A_single": base, "B_hrm": hrm, "K_distill": arms["K_distill"]})
    monkeypatch.setattr(finals3, "TEACHER", hrm)
    monkeypatch.setattr(finals3, "ALL", {"proof_resolve": panel, "cumsum3": {"train": TaskSpec("cumsum3", 5), "eval": {"dev": TaskSpec("cumsum3", 5)}}})
    monkeypatch.setattr(finals3, "PRINCIPAL", ("proof_resolve",))
    monkeypatch.setattr(finals3, "MECH_FAMILIES", ("proof_resolve",))
    monkeypatch.setattr(finals3, "TIER1_FAMILIES", ("proof_resolve",))
    out = tmp_path / "finals"
    fr = finals3.freeze(run / "state.json", out)
    assert "A_cpu" in fr["arms"] and fr["arms"]["K_distill"]["teacher"] == "B_hrm"
    assert fr["arms"]["A_single_last"]["families"] == ["proof_resolve"]
    finals3.run(out, [11, 23], 1, time.time() + 3600)
    rows = [json.loads(l) for l in (out / "final_runs.jsonl").read_text().splitlines()]
    keys = [r["key"] for r in rows]
    assert keys.index("B_hrm|proof_resolve|11") < keys.index("K_distill|proof_resolve|11")
    # tier 1 (core arms on tier-1 families, every seed) completes before any tier-2 cell
    t1 = [i for i, k in enumerate(keys)
          if finals3._tier1(k.split("|")[0], fr["arms"][k.split("|")[0]]) and k.split("|")[1] in finals3.TIER1_FAMILIES]
    t2 = [i for i in range(len(keys)) if i not in t1]
    assert t1 and t2 and max(t1) < min(t2)
    assert not any(k.startswith("A_single_last|cumsum3") for k in keys)
    assert any(r.get("mech") for r in rows if r["arm"] == "B_hrm")
    assert list((out / "checkpoints").glob("*__s11.pt"))
    monkeypatch.setattr(finals3, "run_multitask", finals3.run_multitask)
    res = analysis3.analyse(rows)
    assert res["contrasts"]["P1 source decomposition (vs A)"]["B_hrm - A_single"]["proof_resolve"]["n_seeds"] == 2
    assert "proof_resolve" in res["retention_K_of_B"]
    assert "B_hrm|proof_resolve" in res["table"] and "mech" in res["table"]["B_hrm|proof_resolve"]
    assert analysis3.markdown(res)
    (out / "analysis.json").write_text(json.dumps(res))
    m = export3.build(run / "state.json", run / "dg", out / "analysis.json", out / "freeze.json")
    assert m["kinds"].get("final_evidence") == 1 and m["kinds"].get("hypothesis_cycle") == 2
