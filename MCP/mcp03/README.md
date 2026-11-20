# MCP-03: compact context packets from code-repository MCPs

MCP-03 moves the MCP-02 routers from synthetic catalogs to real repositories, using SWE-bench tasks. Each task gives an issue text and a repository at the instance's base commit. A small local router explores the repository through MCP tools and hands a frontier agent a token-budgeted **packet**: a ranked file map plus source snippets. The gold patch's files and functions are the hidden target. The judge reports how much of that target the packet contains and how many tokens the packet costs.

## Benchmark

| Split | Source | Repositories | Instances |
| --- | --- | --- | ---: |
| train | SWE-bench test minus Lite | django, sympy | 160 (100 + 60) |
| dev | SWE-bench Lite | django, sympy | 32 |
| test | SWE-bench Lite | scikit-learn, matplotlib, pytest, sphinx, astropy, requests, pylint, xarray, seaborn, flask | 109 |

The split is grouped by repository. No test repository contributes training data, so the test measures transfer to unseen codebases. Snapshots come from bare clones via `git ls-tree` and `git cat-file`, so nothing is checked out, and symbols are parsed with `ast` and cached per blob. Gold targets come from the patch: changed files, plus the innermost function, method or class around each changed line of the old file. A module-level edit maps to the module.

## MCP surface

| Tool | Arms | Result |
| --- | --- | --- |
| `code_search(query)` | search, hybrid, layered_ldt | BM25 over symbol documents: path, name, signature, docstring and body identifiers; paginated |
| `index_children(node_id)` | layered, hybrid, layered_ldt | Directories, modules, classes, methods, with counts, sketches and path flags |
| `read_symbol(uri)` | all | Up to 80 source lines of one symbol |
| `read_outline(path)` | all | A module's symbols with signatures |
| `candidate_refine(node_id, exclude, predicate)` | layered_ldt | Certified exclusion of `test_path` or `doc_path` subtrees through `certify_and_apply_verified` |

The routers are MCP-02's, run unchanged behind the same observation boundary. The observation holds the salient issue terms (code identifiers, dotted names, paths, traceback functions) and the frontier of candidates. Controls are the identifier-match rules, BM25, sham random and the oracle bound. The non-agentic baseline is SWE-bench's BM25 retrieval: whole files ranked by BM25 of the issue and packed until the budget (2K, 4K, 13K and 27K tokens).

## Metrics

- **Localisation:** file recall, all gold files present, and symbol recall. A class snippet covers the methods it contains.
- **Tokens:**
  - packet tokens under a named tokenizer (Qwen2.5-0.5B-Instruct `tokenizer.json`, pinned revision);
  - tokens the router read locally across all tool results;
  - router tokens (hashed encoder positions);
  - gold-file and whole-repository sizes for reference.
- **Compute:** router FLOPs (FlopCounterMode times candidates), router milliseconds per decision, tool calls, certified exclusions and their metadata checks.
- **Memory:** process peak RSS during training (training receipts), and inference peak RSS around a 32-candidate forward (`MCP.mcp02.run memprobe`).

## Run

```bash
export MCP03_TOKENIZER=PATH/TO/tokenizer.json   # exact counts; otherwise ceil(bytes/4), labelled
python -m MCP.mcp03.run dataset  --config MCP/mcp03/configs/swebench_lite.json --split train --out WORK/train.json.gz
python -m MCP.mcp03.run evaluate --config MCP/mcp03/configs/swebench_lite.json --out WORK/run [--checkpoints CKPT]
python -m MCP.mcp03.run summarize --run WORK/run
python -m MCP.mcp03.run replay --run WORK/run --every 5
```

`work_dir` in the config must contain `data/*.jsonl` (converted from the SWE-bench parquet files) and `repos/<owner>__<name>` bare clones.

## Goose integration

- [`goblin_server.py`](goblin_server.py) is an MCP stdio server exposing `compact_context(task, repo_path, budget_tokens, ...)` and `read_symbol`. It reads a live working tree. Goose loads it as an extension with `--with-extension "python -m MCP.mcp03.goblin_server"`.
- [`../goose_gdk_app`](../goose_gdk_app/README.md) is a Rust app on the Goose Development Kit (`goose-agent` 0.1.0-alpha.14). It is a state machine with two steps:
  - an `Operation` that contributes the goblin packet as a prompt part;
  - an `InferenceRunner` that calls an OpenAI-compatible provider from `goose-providers`.

  With `--no-packet` it runs the same machine without the packet, so the receipt isolates the packet's effect on provider tokens.
