# Data Goblin: Compact Context Packets from Complex MCP Catalogs with Small Looped Routers

The paper is `main.pdf`; its source is `main.tex` and `sections/`. Every table, figure and quoted number is generated from the committed result directories:

```bash
python papers/mcp_data_goblin/make_tables.py      # tables/*.tex, tables/numbers.tex, macro_defaults.tex, numbers.json
python papers/mcp_data_goblin/make_figures.py     # figures/*.pdf (needs matplotlib)
cd papers/mcp_data_goblin && pdflatex main.tex && pdflatex main.tex
```

| Source | Used for |
| --- | --- |
| `MCP/results/mcp03/swebench_lite` | SWE-bench Lite packets, baselines, router receipts, memory probes |
| `MCP/results/mcp03/gdk_demo` | goblin-gdk with a local model, with and without packets |
| `MCP/results/mcp02/pilot_gpu`, `pilot_cpu` | Synthetic multi-server catalogs: success, sweeps, receipts |
| `MCP/results/mcp02/llm_pilot` | Local LLM as an end-to-end MCP controller |

The code is in `MCP/mcp02`, `MCP/mcp03` and `MCP/goose_gdk_app`. Episode tapes replay without torch through `python -m MCP.mcp02.run replay` and `python -m MCP.mcp03.run replay`.
