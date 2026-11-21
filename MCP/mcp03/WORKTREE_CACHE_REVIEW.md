# MCP-03 live working-tree consistency review

An exact-code retrieval gym can use a pinned snapshot per task. The **live Goose integration** instead allows agents to edit a working tree between `compact_context` and `read_symbol` calls. Its previous `_CACHE` mapped only `repo_path` to a snapshot, leaving code and AST symbols stale for the entire server lifetime after edits.

This review patch gives cached snapshots a local revision signature from the same Python paths `snapshot_from_dir` indexes, incorporating path, size, mtime/ctime nanoseconds and inode. The server rebuilds both snapshot and index after normal edits/additions/deletions, refuses an observed concurrent edit during reconstruction, returns `worktree_revision` on MCP results, and propagates `CodeEnv` read errors through `isError`.

**Scope:** This is a practical freshness check for normal developer workflows. Stat identity is not a cryptographic content proof, and two edits hidden behind identical metadata may evade it. For a science-grade source lock, hash the exact file contents and test the snapshot/trace bytes independently. The patched code remains offline and standard-library only.

**Evaluation boundaries:** `MCP-03`'s 109 SWE-bench Lite test instances are repository-disjoint from training and represent retrieval evaluation, while the 26-task end-to-end Qwen repair run resolved 1 patch in its compact context arm. Those figures do not become additional model outcomes through this engineering patch.

Reproduce:

```sh
python -m pytest -q MCP/test_mcp03_worktree.py MCP/test_mcp03.py
```

The new tests mutate source within a single persistent service instance, add and delete symbols, require correct MCP error flags, and check ignored directory freshness behavior. No GPU or live model endpoint is required.
