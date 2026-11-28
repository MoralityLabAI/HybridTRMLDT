import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research_gym.trm_skill_gym_v3 import distill as D

ARCHIVE = D.DEFAULT_ARCHIVE


@pytest.fixture(scope="module")
def records():
    return D.build_records([("gym01", ARCHIVE)])


def _write(tmp_path, rows, name="a.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return p


def test_input_independent_of_own_targets(records, tmp_path):
    rows = D.load_archive(ARCHIVE)
    base = {r["id"]: r["input"] for r in records}
    for i in (0, 40, 77):
        mod = copy.deepcopy(rows)
        t = mod[i]
        t["rung0"] = {"dev_exact": 0.123}
        t["rung1"] = {"dev_exact": 0.456}
        t["decision"] = "PERTURBED"
        t["status"] = "PERTURBED"
        t["measured_dev_exact"] = 0.789
        t["failure_diagnosis"] = "LEAK-failure"
        t["round_diagnosis"] = "LEAK-round"
        new = {r["id"]: r for r in D.build_records([("gym01", _write(tmp_path, mod, f"m{i}.jsonl"))])}
        rid = f"gym01/{t['arm']}/{t['skill']}/r{t['round']}/p{t['proposal_index']}"
        assert new[rid]["input"] == base[rid]
        assert "LEAK" not in json.dumps(new[rid]["input"])
        assert new[rid]["target"]["rung0_dev_exact"] == 0.123
        # nothing at or before this round of the same arm sees the perturbation
        for r in rows:
            if r["arm"] == t["arm"] and r["round"] <= t["round"]:
                rr = f"gym01/{r['arm']}/{r['skill']}/r{r['round']}/p{r['proposal_index']}"
                assert new[rr]["input"] == base[rr]


def test_failure_modes_only_from_earlier_rounds(records):
    for r in records:
        for m in r["input"]["known_failure_modes"]:
            assert m["round"] < r["input"]["round"]
        for h in r["input"]["prior_results"]:
            assert h["round"] < r["input"]["round"]


def test_missing_outcomes_stay_none(records):
    assert len(records) == 108
    rung0_missing = [r for r in records if r["target"]["rung0_dev_exact"] is None]
    rung1_missing = [r for r in records if r["target"]["rung1_dev_exact"] is None]
    assert len(rung0_missing) == 7 and len(rung1_missing) == 72
    assert {r["target"]["status"] for r in rung0_missing} <= {"rejected_param_bound", "no_change"} | {"ok"}
    assert sum(r["target"]["status"] != "ok" for r in records) == 7
    assert all(r["target"]["rung1_dev_exact"] is None for r in records if r["target"]["decision"] == "raced_out_rung0")
    for r in records:
        rec = r["target"]["verification_receipt"]
        assert len(rec["source_row_sha256"]) == 64 and len(rec["source_file_sha256_lf"]) == 64
        assert "predicted_resource_use" in r["input"] and "task_objective" in r["input"]


def test_extra_archive_adds_records(records, tmp_path):
    rows = D.load_archive(ARCHIVE)[:12]
    p = _write(tmp_path, rows)
    both = D.build_records([("gym01", ARCHIVE), ("extra0", p)])
    assert len(both) == 108 + 12
    assert len({r["id"] for r in both}) == len(both)
    with pytest.raises(ValueError):
        D.build_records([("gym01", ARCHIVE), ("gym01", p)])


def test_holdout_partitions_disjoint(records):
    schemes, fam = D.make_schemes(records)
    n = len(records)
    skills = np.array([r["skill"] for r in records])
    assert set(schemes) == {"research_stopping", "leave_skill_out", "leave_intervention_out"}
    for name, folds in schemes.items():
        tests = np.concatenate([f[2] for f in folds])
        assert len(tests) == len(set(tests)) <= n
        for _, tr, te in folds:
            assert not set(tr) & set(te)
            assert len(tr) + len(te) == n
    assert len(np.concatenate([f[2] for f in schemes["leave_skill_out"]])) == n
    assert len(np.concatenate([f[2] for f in schemes["leave_intervention_out"]])) == n
    for s, tr, te in schemes["leave_skill_out"]:
        assert not set(skills[tr]) & set(skills[te])
    assert len(schemes["leave_skill_out"]) == 6
    assert any(f[0] == "visits" for f in schemes["leave_intervention_out"])
    fams = {f[0] for f in schemes["leave_intervention_out"]}
    assert "verifier_visits" in fams and "hidden" in fams


def test_pilot_reproduction(records):
    out = D.pilot_reproduction(records)
    assert round(out["rung0"]["mae_pp"], 2) == 10.46
    assert round(out["rung1"]["mae_pp"], 2) == 5.40
    assert round(out["rung0"]["pair_accuracy"], 3) == 0.406
    from research_gym.trm_skill_gym_v2.student import run
    ref = run(D.load_archive(ARCHIVE))["rungs"]
    for rg in D.RUNGS:
        assert abs(out[rg]["mae_pp"] - 100 * ref[rg]["ridge"]["mae"]) < 1e-6
        assert abs(out[rg]["bias_pp"] - 100 * ref[rg]["ridge"]["bias"]) < 1e-6


def test_experiment_deterministic(records):
    sub = [r for r in records if r["arm"] != "random_mutation"][:36]
    a = D.run_experiment(sub, gru_epochs=1, n_boot=10)[0]
    b = D.run_experiment(sub, gru_epochs=1, n_boot=10)[0]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_cli_writes_results_and_export(tmp_path):
    o = tmp_path / "o"
    assert D.main(["--out", str(o), "--gru-epochs", "2", "--n-boot", "20"]) == 0
    for name in ("results.json", "distill_records.jsonl"):
        assert b"\r\n" not in (o / name).read_bytes()
    res = json.loads((o / "results.json").read_text(encoding="utf-8"))
    assert res["n_records"] == 108 and res["sources"][0]["sha256_lf"]
    assert round(res["pilot_reproduction"]["rung0"]["mae_pp"], 2) == 10.46
    for s in ("research_stopping", "leave_skill_out", "leave_intervention_out"):
        for rg in D.RUNGS:
            assert "ridge_pilot" in res["schemes"][s]["rungs"][rg]["models"]
    lines = [json.loads(x) for x in (o / "distill_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 108
    req = ["schema_version", "candidate_id", "architecture_family", "mock", "weights_loaded", "training_splits",
           "training_split_hashes", "dataset_manifest_sha256", "maximum_loops", "resource_summary"]
    for r in lines:
        assert all(k in r for k in req) and r["schema_version"] == "1.0"
        assert set(r["split"]) == {"research_stopping", "leave_skill_out", "leave_intervention_out"}
        assert "rung0" not in r["input"] and "decision" not in r["input"]
