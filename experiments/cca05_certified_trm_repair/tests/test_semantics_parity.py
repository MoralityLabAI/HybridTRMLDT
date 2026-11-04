"""The torch-free semantics equal rollout.py and the benchmark's region sampler."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from research_gym.benchmarks.gaming_vs_improvement_bench import (  # noqa: E402
    GamingBenchmarkConfig,
    build_region_heldout_examples,
)
from research_gym.neural import rollout as RO  # noqa: E402

import semantics as M  # noqa: E402


def test_verifier_utility_oracle_and_fallback_equal_rollout_everywhere():
    for sc in M.SCENARIOS:
        for s in M.ENV.all_states():
            for h in range(1, M.HORIZON + 1):
                ex = RO.StoryExample("p", sc, s, h, "p", "p")
                for a in M.ACTIONS:
                    assert M.admissible(sc, s, h, a) == RO.action_environment_sound(M.ENV, ex, a)
                    assert M.utility(sc, s, h, a) == RO.action_value(M.ENV, ex, a)
                assert M.oracle_action(sc, s, h) == RO.oracle_action(M.ENV, ex)
                assert M.original_fallback_action(sc, s, h) == RO.state_conditioned_fallback(M.ENV, ex)


@pytest.mark.parametrize("seed", [211, 223, 227])
def test_region_sampling_equals_benchmark(seed):
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "gaming_vs_improvement_v2.json"
    cfg = GamingBenchmarkConfig.from_registration(json.loads(cfg_path.read_text(encoding="utf-8")))
    bench = build_region_heldout_examples(cfg, seed=seed)
    mine = M.sampled_regions(seed)
    for name, rows in bench.items():
        assert [(r.scenario, r.state) for r in rows] == mine[name]
