"""Generated platformer repository contracts and private-label separation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from MCP.mcp03.agent import run_code_episode, make_code_control
from MCP.mcp03.env import CodeEnv
from MCP.mcp03.platformer_gen import generate
from MCP.mcp03.repo import SnapshotStore, gold_symbols, load_instances


@pytest.fixture(scope="module")
def small_corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("platformer-cases") / "campaign"
    return out, generate(out, seed=113, train=2, dev=1, test=1)


def test_generator_creates_git_bare_clones_and_distinct_roots(small_corpus):
    out, manifest = small_corpus
    assert manifest["cases"] == {"train": 2, "dev": 1, "test": 1}
    groups = [set(value) for value in manifest["roots"].values()]
    assert not any(a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    store = SnapshotStore(out / "repos", out / "cache")
    for split in ("train", "dev", "test"):
        rows = load_instances(out / "data" / f"{split}.jsonl")
        assert len(rows) == manifest["cases"][split]
        for row in rows:
            snap = store.snapshot(row["repo"], row["base_commit"])
            gold = gold_symbols(snap, row["patch"])
            assert len(gold["files"]) == 1 and gold["symbols"]
            assert not any(word in row["problem_statement"] for word in ("gold_symbols", "patch_sha256", "diff --git"))
            assert snap.files and snap.symbols


def test_config_fresh_splits_and_equal_evaluation_budgets(small_corpus):
    out, manifest = small_corpus
    read = json.loads((out / "config_readaware.json").read_text(encoding="utf-8"))
    binary = json.loads((out / "config_binary.json").read_text(encoding="utf-8"))
    eval_cfg = json.loads((out / "config_eval.json").read_text(encoding="utf-8"))
    assert read["dataset"]["utility_targets"] is True
    assert binary["dataset"]["utility_targets"] is False
    assert eval_cfg["evaluation"]["max_tool_calls"] == [24]
    assert eval_cfg["evaluation"]["budgets"] == [1024]
    assert eval_cfg["evaluation"]["composed_packets"] is False
    assert set(eval_cfg["models"]) == {"looped_p4_d64_readaware", "looped_p4_d64_binary"}
    assert all(set(read["splits"][s]["repos"]) == set(manifest["roots"][s]) for s in manifest["roots"])


def test_generated_task_retrieval_uses_public_issue_only(small_corpus):
    out, _manifest = small_corpus
    row = load_instances(out / "data" / "test.jsonl")[0]
    snap = SnapshotStore(out / "repos", out / "cache").snapshot(row["repo"], row["base_commit"])
    gold = gold_symbols(snap, row["patch"])
    env = CodeEnv(snap, {}, gold, "search")
    obs_seen = []

    class ObservePolicy:
        name = "observe-lexical"
        family = "test"
        uses_private_oracle = False

        def reset(self, episode_key):
            self.policy = make_code_control("lexical_bm25")
            self.policy.reset(episode_key)

        def score(self, obs):
            obs_seen.append(obs)
            return self.policy.score(obs)

        def last_cost(self):
            return self.policy.last_cost()

        def describe(self):
            return {"name": self.name, "family": self.family, "uses_private_oracle": False}

    result = run_code_episode(env, row["problem_statement"], policy=ObservePolicy(),
                              max_tool_calls=8, budgets=(1024,))
    assert obs_seen and result["tool_calls"] <= 8
    assert result["packets"]["1024"]["tokens"] <= 1024
    for obs in obs_seen:
        assert set(obs) >= {"task", "candidates", "remaining_calls"}
        assert "patch" not in obs["task"] and "gold" not in obs
        assert "utility_values" not in obs


def test_generator_seed_reproducibly_selects_families(tmp_path):
    first = generate(tmp_path / "first", seed=89, train=1, dev=1, test=1)
    second = generate(tmp_path / "second", seed=89, train=1, dev=1, test=1)
    for split in ("train", "dev", "test"):
        def receipts(data):
            return [(r["family"], r["base_commit"], r["patch_sha256"], r["gold_digest"])
                    for r in data["gold_receipts"][split]]
        assert receipts(first) == receipts(second)


def test_hard_corpus_has_three_verified_target_symbols(tmp_path):
    out = tmp_path / "hard"
    manifest = generate(out, seed=137, train=1, dev=1, test=1, difficulty="hard")
    assert manifest["difficulty"] == "hard"
    for split, rows in manifest["gold_receipts"].items():
        assert len(rows) == 1
        assert len(rows[0]["families"]) == 3
        assert len(rows[0]["gold_symbols"]) >= 3
    cfg = json.loads((out / "config_eval.json").read_text(encoding="utf-8"))
    assert "oracle_upper_bound" in cfg["evaluation"]["controls"]


def test_shifted_holdout_renames_test_symbols_and_retains_gold(tmp_path):
    from MCP.mcp03.platformer_gen import HOLDOUT_SYMBOLS
    out = tmp_path / "shifted"
    manifest = generate(out, seed=173, train=1, dev=1, test=1,
                        difficulty="hard", holdout_symbol_names=True)
    assert manifest["holdout_symbol_names"] is True
    store = SnapshotStore(out / "repos", out / "cache")
    train = load_instances(out / "data" / "train.jsonl")[0]
    test = load_instances(out / "data" / "test.jsonl")[0]
    old = store.snapshot(train["repo"], train["base_commit"])
    new = store.snapshot(test["repo"], test["base_commit"])
    old_names = {s.qualname for s in old.all_symbols()}
    new_names = {s.qualname for s in new.all_symbols()}
    renamed = set(HOLDOUT_SYMBOLS.values())
    assert not old_names.intersection(renamed)
    assert len(new_names.intersection(renamed)) >= 1
    assert gold_symbols(new, test["patch"])["files"]
    assert manifest["gold_receipts"]["test"][0]["surface_shift"] is True
    assert manifest["gold_receipts"]["train"][0]["surface_shift"] is False


def test_matched_ablation_config_has_same_training_split_and_four_arms(tmp_path):
    out = tmp_path / "matched"
    generate(out, seed=179, train=1, dev=1, test=1, difficulty="easy")
    cfg = json.loads((out / "config_matched.json").read_text(encoding="utf-8"))
    assert cfg["dataset"]["utility_targets"] is True
    assert set(cfg["models"]) == {
        "looped_binary_matched", "looped_shared_utility",
        "looped_dual_utility", "looped_dual_shuffled"}
    assert cfg["models"]["looped_binary_matched"]["config"]["use_horizon"]
    assert cfg["models"]["looped_shared_utility"]["config"]["single_utility"]
    assert cfg["models"]["looped_dual_shuffled"]["shuffle_utilities_only"]
    assert cfg["evaluation"]["max_tool_calls"] == [24]
    assert cfg["evaluation"]["budgets"] == [1024]
