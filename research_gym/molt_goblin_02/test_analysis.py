from .analysis import analyse, pareto, to_markdown
from .factory import MoltConfig
from .gym import TaskSpec
from .interventions import subspace_panel
from .train import train_and_eval


def test_analysis_end_to_end():
    spec = TaskSpec("cumsum", 5)
    runs = []
    for arm, cfg in (("prefix", MoltConfig(width=16, loops=2, steps=16)),
                     ("hrm", MoltConfig(width=16, loops=2, steps=16, routing="hrm", n_types=2, halt="utility"))):
        for seed in (1, 2):
            r = train_and_eval(cfg, "cumsum", spec, {"dev": spec, "ood_long": TaskSpec("cumsum", 7)}, seed, n_eval=32,
                               extrap_loops=4, return_model=True)
            model = r.pop("_model")
            r["subspace"] = subspace_panel(model, spec, 3, 4, ranks=(2, 4), n=32)
            runs.append({"arm": arm, "kind": "x", **r})
    res = analyse(runs)
    assert "hrm|cumsum" in res["paired_vs_reference"]
    assert res["pareto_front"]
    row = res["table"]["hrm|cumsum"]
    assert "routing_entropy" in row and "halt_exact" in row and "subspace" in row
    assert "| hrm |" in to_markdown(res)


def test_pareto():
    pts = {"a": {"quality": 0.5, "cpu_s": 1, "bytes": 1}, "b": {"quality": 0.4, "cpu_s": 2, "bytes": 2},
           "c": {"quality": 0.6, "cpu_s": 3, "bytes": 1}}
    assert sorted(pareto(pts)) == ["a", "c"]
