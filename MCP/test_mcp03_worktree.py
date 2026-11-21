"""Live-worktree freshness and error-channel regressions for the Goose MCP server."""
from __future__ import annotations

from pathlib import Path

from MCP.mcp03.goblin_server import GoblinService, _worktree_revision


def test_read_symbol_tracks_edits_new_files_and_removal(tmp_path: Path):
    root = tmp_path / "work"
    root.mkdir()
    mod = root / "core.py"
    mod.write_text("def target():\n    return 'before'\n", encoding="utf-8")
    svc = GoblinService()
    args = {"repo_path": str(root), "uri": "core.py::target"}

    first = svc.call("read_symbol", args)
    assert first["isError"] is False
    assert "'before'" in first["structuredContent"]["source"]
    rev_before = first["structuredContent"]["worktree_revision"]

    # Same filename and same symbol; the MCP source must match new working state.
    mod.write_text("def target():\n    return 'after'\n", encoding="utf-8")
    second = svc.call("read_symbol", args)
    assert second["isError"] is False
    assert "'after'" in second["structuredContent"]["source"]
    assert second["structuredContent"]["worktree_revision"] != rev_before

    extra = root / "added.py"
    extra.write_text("def new_symbol():\n    return 9\n", encoding="utf-8")
    added = svc.call("read_symbol", {"repo_path": str(root), "uri": "added.py::new_symbol"})
    assert added["isError"] is False
    assert "return 9" in added["structuredContent"]["source"]

    extra.unlink()
    deleted = svc.call("read_symbol", {"repo_path": str(root), "uri": "added.py::new_symbol"})
    assert deleted["isError"] is True
    assert "error" in deleted["structuredContent"]


def test_mcp_errors_are_propagated_not_disguised_as_success(tmp_path: Path):
    root = tmp_path / "work"
    root.mkdir()
    (root / "core.py").write_text("def ok():\n    pass\n", encoding="utf-8")
    service = GoblinService()
    error = service.call("read_symbol", {"repo_path": str(root), "uri": "core.py::missing"})
    assert error["isError"] is True
    assert "error" in error["structuredContent"]
    good = service.call("read_symbol", {"repo_path": str(root), "uri": "core.py::ok"})
    assert good["isError"] is False


def test_revision_excludes_ignored_directories_and_tracks_replacement(tmp_path: Path):
    root = tmp_path / "work"
    root.mkdir()
    py = root / "core.py"
    py.write_text("a = 1\n", encoding="utf-8")
    rev1 = _worktree_revision(root)
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "opaque.py").write_text("a = 999\n", encoding="utf-8")
    assert _worktree_revision(root) == rev1
    temp = root / "temp.py"
    temp.write_text("a = 2\n", encoding="utf-8")
    temp.replace(py)
    assert _worktree_revision(root) != rev1
