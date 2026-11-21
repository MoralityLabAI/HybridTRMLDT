"""MCP-03 contracts on a throwaway git repository: snapshots, gold targets, tools, certification, packets."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from MCP.mcp03.agent import (CODE_CONTROLS, bm25_file_baseline, make_code_control, run_code_episode,
                             salient_terms, symbolic_code_exclusions)
from MCP.mcp03.env import CodeEnv
from MCP.mcp03.repo import SnapshotStore, gold_symbols, parse_symbols, patch_targets, snapshot_from_dir
from MCP.mcp03.tokens import count_tokens

CORE = textwrap.dedent('''\
    """Core arithmetic."""


    def add(a, b):
        """Add two numbers."""
        return a + b


    class Ledger:
        """Running totals."""

        def __init__(self):
            self.total = 0

        def post(self, amount):
            self.total = add(self.total, amount)
            return self.total
    ''')
FIXED = CORE.replace("self.total = add(self.total, amount)", "self.total = add(self.total, round(amount, 2))")
TEST = "from pkg.core import Ledger\n\n\ndef test_post():\n    assert Ledger().post(1) == 1\n"
DOC = "def build_docs():\n    return 'docs'\n"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("mcp03")
    work = root / "work"
    (work / "pkg").mkdir(parents=True)
    (work / "tests").mkdir()
    (work / "docs").mkdir()
    (work / "pkg" / "__init__.py").write_text("")
    (work / "pkg" / "core.py").write_text(CORE)
    (work / "tests" / "test_core.py").write_text(TEST)
    (work / "docs" / "conf_tools.py").write_text(DOC)
    _git(work, "init", "-q")
    _git(work, "-c", "user.name=t", "-c", "user.email=t@example.com", "add", "-A")
    _git(work, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "base")
    commit = _git(work, "rev-parse", "HEAD").strip()
    (work / "pkg" / "core.py").write_text(FIXED)
    patch = _git(work, "diff")
    (work / "pkg" / "core.py").write_text(CORE)
    repos = root / "repos"
    repos.mkdir()
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(repos / "acme__ledger")], check=True)
    store = SnapshotStore(repos, root / "cache")
    snap = store.snapshot("acme/ledger", commit)
    return {"work": work, "snap": snap, "patch": patch, "gold": gold_symbols(snap, patch)}


def test_symbols_and_gold_targets(repo):
    syms = {s.qualname: s for s in parse_symbols("pkg/core.py", CORE)}
    assert {"<module>", "add", "Ledger", "Ledger.__init__", "Ledger.post"} <= set(syms)
    assert syms["Ledger.post"].kind == "method" and syms["add"].kind == "function"
    assert set(patch_targets(repo["patch"])) == {"pkg/core.py"}
    assert repo["gold"] == {"files": ["pkg/core.py"], "symbols": ["pkg/core.py::Ledger.post"]}
    assert set(repo["snap"].files) == {"pkg/__init__.py", "pkg/core.py", "tests/test_core.py", "docs/conf_tools.py"}


def test_tools_and_errors(repo):
    env = CodeEnv(repo["snap"], {}, repo["gold"], "layered_ldt")
    hits = env.call("code_search", {"query": "Ledger post amount"})["structuredContent"]["results"]
    assert hits[0]["uri"] in ("pkg/core.py::Ledger.post", "pkg/core.py::Ledger")
    root = env.call("index_children", {"node_id": "d:"})["structuredContent"]["children"]
    assert {r["path"] for r in root} == {"pkg", "tests", "docs"}
    src = env.call("read_symbol", {"uri": "pkg/core.py::Ledger.post"})["structuredContent"]["source"]
    assert "def post" in src
    assert env.call("read_symbol", {"uri": "nope"})["isError"]
    assert env.call("index_children", {"node_id": "d:missing"})["isError"]


def test_certified_path_exclusions(repo):
    env = CodeEnv(repo["snap"], {}, repo["gold"], "layered_ldt")
    rows = env.call("index_children", {"node_id": "d:"})["structuredContent"]["children"]
    proposals = dict(symbolic_code_exclusions(rows))
    assert proposals == {"d:tests": "test_path", "d:docs": "doc_path"}
    bad = env.call("candidate_refine", {"node_id": "d:", "exclude": ["d:pkg"], "predicate": "test_path"})
    assert bad["structuredContent"]["excluded"] == []
    ok = env.call("candidate_refine", {"node_id": "d:", "exclude": ["d:tests"], "predicate": "test_path"})
    assert ok["structuredContent"]["excluded"] == ["d:tests"]
    kept = env.call("index_children", {"node_id": "d:"})["structuredContent"]["children"]
    assert {r["path"] for r in kept} == {"pkg", "docs"}
    assert env.private["certified_exclusions_containing_gold"] == 0


@pytest.mark.parametrize("arm", ["search", "layered", "hybrid", "layered_ldt"])
def test_controls_build_budgeted_packets(repo, arm):
    issue = "Ledger.post accumulates float error; `post` should round the amount before calling add."
    for name in CODE_CONTROLS:
        env = CodeEnv(repo["snap"], {}, repo["gold"], arm)
        ep = run_code_episode(env, issue, policy=make_code_control(name), max_tool_calls=12, budgets=(64, 512))
        for b, packet in ep["packets"].items():
            assert packet["tokens"] <= int(b)
        if name == "oracle_upper_bound":
            assert ep["packets"]["512"]["grade"]["file_recall"] == 1.0
            assert ep["packets"]["512"]["grade"]["symbol_recall"] == 1.0


def test_composed_packet_uses_file_search(repo):
    issue = "Ledger post should round the amount"
    env = CodeEnv(repo["snap"], {}, repo["gold"], "layered")
    ep = run_code_episode(env, issue, policy=make_code_control("sham_random"), max_tool_calls=3, budgets=(256,),
                          compose=True)
    assert ep["packets"]["c256"]["grade"]["file_recall"] == 1.0 and ep["packets"]["c256"]["tokens"] <= 256
    assert ep["tool_calls"] == 4 and ep["environment_stats"]["search_calls"] >= 1


def test_tape_replay(repo):
    issue = "Ledger.post should round the amount"
    env = CodeEnv(repo["snap"], {}, repo["gold"], "layered_ldt")
    ep = run_code_episode(env, issue, policy=make_code_control("lexical_bm25"), max_tool_calls=10, budgets=(256,))
    again = run_code_episode(CodeEnv(repo["snap"], {}, repo["gold"], "layered_ldt"), issue,
                             policy=make_code_control("sham_random"), max_tool_calls=10, budgets=(256,),
                             tape=ep["tape"])
    assert again["results_sha256"] == ep["results_sha256"] and again["packets"] == ep["packets"]


def test_baseline_terms_and_tokens(repo):
    env = CodeEnv(repo["snap"], {}, repo["gold"], "search")
    base = bm25_file_baseline(env, "Ledger post amount rounding", (16, 4096))
    assert base["packets"]["4096"]["grade"]["file_recall"] == 1.0 and base["packets"]["16"]["tokens"] <= 16
    terms = salient_terms("Calling `Ledger.post(x)` in pkg/core.py raises; see post_amount and FooBar")
    assert {"Ledger", "post", "core", "post_amount", "FooBar"} <= set(terms)
    assert count_tokens("abcd" * 10) >= 1


def test_goblin_server_over_stdio(repo):
    from MCP.transport import StdioClient
    snap = snapshot_from_dir(repo["work"])
    assert "pkg/core.py" in snap.files
    with StdioClient(command=[sys.executable, "-u", "-m", "MCP.mcp03.goblin_server"], timeout=60,
                     max_message_bytes=8 * 1024 * 1024) as client:
        client.initialize()
        assert {t["name"] for t in client.tools()} == {"compact_context", "read_symbol"}
        result = client.call("compact_context", {"task": "Ledger.post should round the amount",
                                                 "repo_path": str(repo["work"]), "budget_tokens": 256})
        payload = result["structuredContent"]
        assert 0 < payload["packet_tokens"] <= 256 and "pkg/core.py" in payload["packet"]
