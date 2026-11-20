"""Generate the paper's LaTeX tables and headline numbers from committed MCP-02/MCP-03 results.

    python papers/mcp_data_goblin/make_tables.py

Reads MCP/results/mcp02/*/summary.json, model_receipts.json and memory probes,
MCP/results/mcp03/*/summary.json and the GDK demo receipts, then writes
papers/mcp_data_goblin/tables/*.tex and numbers.json. Every number in the
paper's tables comes from these files.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tables"
R2 = ROOT / "MCP" / "results" / "mcp02"
R3 = ROOT / "MCP" / "results" / "mcp03"


def tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt(x, d=2):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def table(name: str, header: list[str], rows: list[list], caption: str, label: str, align: str | None = None):
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(tex(fmt(c)) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mcp02_tables(numbers: dict) -> None:
    for run_name in ("pilot_cpu", "pilot_gpu"):
        run = R2 / run_name
        summary = load(run / "summary.json")
        if summary is None:
            continue
        receipts = load(run / "model_receipts.json") or {}
        groups = summary["groups"]
        main = [g for g in groups if g["experiment"] == "main"]
        cells = sorted({(g["catalog_size"], g["arm"], g["layout"]) for g in main})
        policies = sorted({g["policy"] for g in main})
        idx = {(g["catalog_size"], g["arm"], g["layout"], g["policy"]): g for g in main}
        rows = []
        for p in policies:
            row = [p]
            for c in cells:
                g = idx.get((*c, p))
                row.append(f"{g['accepted']}/{g['episodes']}" if g else "--")
            rows.append(row)
        header = ["Router"] + [f"{a}{('/' + l) if l else ''}@{s}" for s, a, l in cells]
        table(f"mcp02_{run_name}_success", header, rows,
              f"MCP-02 {run_name.replace('_', ' ')}: accepted episodes per arm and catalog size "
              "(32 held-out test worlds per cell, 40-call budget).", f"tab:mcp02-{run_name}-success",
              "l" + "c" * len(cells))
        # model receipts: parameters, depth, FLOPs, latency, memory
        mem = {}
        for path in (run / "memory").glob("*.mem.json") if (run / "memory").is_dir() else []:
            mem[path.name[:-9]] = load(path)
        ms = {}
        for g in main:
            if g["router_ms_per_decision"] is not None:
                ms.setdefault(g["policy"], []).append(g["router_ms_per_decision"])
        rows = []
        for name, rec in sorted(receipts.items()):
            params = rec.get("parameters") or {}
            if not isinstance(params, dict) or "total" not in params:
                continue
            training = rec.get("training") or {}
            m = mem.get(name, {})
            delta = m.get("forward_peak_rss_delta")
            train_mem = (training.get("memory") or {})
            rows.append([name, rec.get("family"), params.get("total"), params.get("non_embedding"),
                         rec.get("recurrent_depth"), (rec.get("flops_per_candidate") or {}).get("flops"),
                         sum(ms.get(name, [0])) / max(1, len(ms.get(name, []))) if ms.get(name) else None,
                         round(delta / 2**20, 1) if delta else None,
                         round(train_mem["process_peak_rss_bytes"] / 2**20) if train_mem.get("process_peak_rss_bytes") else None,
                         round(train_mem["cuda_max_memory_allocated"] / 2**20) if train_mem.get("cuda_max_memory_allocated") else None,
                         (training.get("dev_ranking") or {}).get("top1_positive")])
        table(f"mcp02_{run_name}_receipts",
              ["Router", "Family", "Params", "Non-emb.", "Depth", "FLOPs/cand.", "ms/decision", "Infer RSS MiB",
               "Train RSS MiB", "Train CUDA MiB", "Dev top-1"], rows,
              f"MCP-02 {run_name.replace('_', ' ')} router receipts: measured parameters, recurrent depth "
              "(block applications), FLOPs per candidate (FlopCounterMode), router latency, peak memory.",
              f"tab:mcp02-{run_name}-receipts")
        depth = [g for g in groups if g["experiment"] == "depth"]
        if depth:
            rows = [[g["policy"], f"{g['arm']}/{g['layout']}", g["depth"], f"{g['accepted']}/{g['episodes']}",
                     g["tool_calls_mean"], (g["router_flops_total"] / max(1, g["episodes"])) / 1e9]
                    for g in sorted(depth, key=lambda g: (g["policy"], g["arm"], g["depth"]))]
            table(f"mcp02_{run_name}_depth", ["Router", "Arm", "Test depth", "Accepted", "Calls/ep", "GFLOPs/ep"],
                  rows, "Recurrent depth changed at test time only (trained depth in the router name).",
                  f"tab:mcp02-{run_name}-depth")
        layout = [g for g in groups if g["experiment"] == "layout"]
        if layout:
            rows = [[g["policy"], g["arm"], g["layout"] or "-", len(g["layout"]) + 1, f"{g['accepted']}/{g['episodes']}",
                     g["tool_calls_mean"], g["tool_result_bytes_mean"] / 1024, g["ldt_excluded"]]
                    for g in sorted(layout, key=lambda g: (g["policy"], g["arm"], len(g["layout"]), g["layout"]))]
            table(f"mcp02_{run_name}_layout", ["Router", "Arm", "Layout", "Depth", "Accepted", "Calls/ep",
                                               "Result KB/ep", "Certified excl."], rows,
                  "Index layout and depth at 1024 resources: deeper facet hierarchies against traversal cost.",
                  f"tab:mcp02-{run_name}-layout")
        best = max(main, key=lambda g: (g["accepted"], -g["tool_calls_mean"]))
        numbers[f"mcp02_{run_name}"] = {"episodes": summary["episodes"], "best_main": [best["policy"], best["arm"],
                                                                                      best["catalog_size"]]}


def mcp03_tables(numbers: dict) -> None:
    run = R3 / "swebench_lite"
    summary = load(run / "summary.json")
    if summary is None:
        return
    rows = []
    for g in summary["groups"]:
        if g["policy"] == "sham_random" and g["arm"] == "layered":
            continue
        for b in ("c512", "c2048", "512", "2048"):
            if b not in g["budgets"]:
                continue
            x = g["budgets"][b]
            label = g["arm"] + (" +map" if b.startswith("c") else "")
            rows.append([g["policy"], label, int(b.lstrip("c")), x["packet_tokens_mean"], x["file_recall_mean"],
                         f"{x['all_files_hit']}/{g['instances']}", x["symbol_recall_mean"],
                         g["tool_calls_mean"], g["retrieval_result_tokens_mean"]])
    for b, x in sorted(summary["bm25_file_baseline"].items(), key=lambda kv: int(kv[0])):
        rows.append(["bm25_whole_files", "direct", int(b), x["tokens_mean"], x["file_recall_mean"],
                     f"{x['all_files_hit']}/{x['instances']}", None, 0, None])
    for k, x in sorted(summary.get("bm25_path_map_baseline", {}).items(), key=lambda kv: int(kv[0])):
        rows.append([f"bm25_path_map_top{k}", "direct", None, x["tokens_mean"], x["file_recall_mean"],
                     f"{x['all_files_hit']}/{x['instances']}", None, 0, None])
    table("mcp03_packets", ["Router", "Arm", "Budget", "Packet tok.", "File recall", "All files", "Symbol recall",
                            "Calls", "Local tok. read"], rows,
          "MCP-03 on 109 SWE-bench Lite instances from ten repositories unseen in training: packet size "
          "(Qwen2.5 tokenizer) against localisation of the gold patch, with the tokens the router read locally.",
          "tab:mcp03-packets")
    numbers["mcp03"] = {"instances": summary["instances"], "context": summary["context"]}
    receipts = load(run / "model_receipts.json") or {}
    mem_dir = run / "memory"
    by_policy = {}
    for g in summary["groups"]:
        by_policy.setdefault(g["policy"], {})[g["arm"]] = g
    rows = []
    for name, rec in sorted(receipts.items()):
        params = rec.get("parameters") or {}
        if not isinstance(params, dict) or "non_embedding" not in params:
            continue
        inf = load(mem_dir / f"{name}.mem.json") or {}
        tcuda = load(mem_dir / f"{name}.train_cuda.json") or {}
        tcpu = load(mem_dir / f"{name}.train_cpu.json") or {}
        ldt = by_policy.get(name, {}).get("layered_ldt") or by_policy.get(name, {}).get("search")
        mib = lambda b: round(b / 2**20) if b else None  # noqa: E731
        rows.append([name, rec.get("family"), params["total"], params["non_embedding"], rec.get("recurrent_depth"),
                     (rec.get("flops_per_candidate") or {}).get("flops"),
                     ldt["router_ms_per_decision"] if ldt else None,
                     mib(inf.get("forward_peak_rss_delta")),
                     mib((tcpu.get("peak_rss_training") or 0) - (tcpu.get("peak_rss_after_dataset_and_torch_load") or 0)),
                     mib(tcuda.get("cuda_max_memory_allocated")),
                     tcuda.get("seconds_per_step"),
                     ldt["budgets"].get("c2048", ldt["budgets"]["2048"])["file_recall_mean"] if ldt else None,
                     ldt["budgets"]["2048"]["symbol_recall_mean"] if ldt else None])
    if rows:
        table("mcp03_receipts", ["Router", "Family", "Params", "Non-emb.", "Depth", "FLOPs/cand.", "ms/dec.",
                                 "Infer +MiB", "Train CPU +MiB", "Train CUDA MiB", "s/step GPU", "File R@2K+map",
                                 "Sym R@2K"], rows,
              "MCP-03 router modules: measured size, recurrent depth, FLOPs per candidate, latency, inference memory "
              "delta around a 32-candidate forward, training memory over 15 CPU / 40 GPU steps (CPU: growth above the "
              "loaded dataset and runtime; GPU: peak allocation), and held-out localisation in the LDT arm.",
              "tab:mcp03-receipts")


def gdk_table(numbers: dict) -> None:
    demo = load(R3 / "gdk_demo" / "receipts.json")
    if not demo:
        return
    rows = [[d["label"], d["used_packet"], d.get("packet_tokens"), d["provider_input_tokens"],
             d["provider_output_tokens"], d["retrieval_seconds"], d["inference_seconds"]] for d in demo["runs"]]
    table("gdk_demo", ["Run", "Packet", "Packet tok.", "Provider in", "Provider out", "Retrieval s", "Inference s"],
          rows, "The goblin-gdk Goose GDK app with and without the compact packet.", "tab:gdk-demo")
    numbers["gdk_demo"] = demo.get("summary", {})


def macros(numbers: dict) -> None:
    """LaTeX macros for headline numbers quoted in the prose."""
    out = []

    def put(name: str, value) -> None:
        out.append(rf"\newcommand{{\{name}}}{{{tex(fmt(value))}}}")

    s3 = load(R3 / "swebench_lite" / "summary.json")
    if s3:
        groups = {(g["arm"], g["policy"]): g for g in s3["groups"]}
        def recall(arm, policy, budget, key="file_recall_mean"):
            g = groups.get((arm, policy))
            return round(g["budgets"][budget][key], 3) if g and budget in g["budgets"] else None
        put("McpThreeInstances", s3["instances"])
        put("BmRouterFileRecallFiveTwelve", recall("search", "lexical_bm25", "512"))
        put("BmComposedFileRecallTwoK", recall("search", "lexical_bm25", "c2048"))
        put("BmComposedSymRecallTwoK", recall("search", "lexical_bm25", "c2048", "symbol_recall_mean"))
        put("OracleLdtComposedFileRecallTwoK", recall("layered_ldt", "oracle_upper_bound", "c2048"))
        put("BmRouterFileRecallTwoK", recall("search", "lexical_bm25", "2048"))
        put("BmRouterSymRecallTwoK", recall("search", "lexical_bm25", "2048", "symbol_recall_mean"))
        put("OracleLdtFileRecallTwoK", recall("layered_ldt", "oracle_upper_bound", "2048"))
        put("OracleLdtSymRecallTwoK", recall("layered_ldt", "oracle_upper_bound", "2048", "symbol_recall_mean"))
        b = s3["bm25_file_baseline"]
        for key, name in (("2048", "Two"), ("4096", "Four"), ("13000", "Thirteen"), ("27000", "TwentySeven")):
            if key in b:
                put(f"BmFilesRecall{name}", round(b[key]["file_recall_mean"], 3))
                put(f"BmFilesTokens{name}", round(b[key]["tokens_mean"]))
        maps = s3.get("bm25_path_map_baseline", {})
        for key, name in (("5", "Five"), ("12", "Twelve"), ("25", "TwentyFive")):
            if key in maps:
                put(f"BmMapRecall{name}", round(maps[key]["file_recall_mean"], 3))
                put(f"BmMapTokens{name}", round(maps[key]["tokens_mean"]))
        put("GoldFilesTokens", round(s3["context"]["gold_files_tokens_mean"]))
        put("RepoPythonFiles", round(s3["context"]["repo_python_files_mean"]))
        learned = [g for g in s3["groups"] if g["family"] not in ("lexical", "deterministic", "sham", "oracle")]
        if learned:
            best = max(learned, key=lambda g: g["budgets"]["2048"]["symbol_recall_mean"])
            put("BestLearnedRouter", best["policy"])
            put("BestLearnedArm", best["arm"].replace("_", " "))
            put("BestLearnedFileRecallTwoK", round(best["budgets"]["2048"]["file_recall_mean"], 3))
            put("BestLearnedSymRecallTwoK", round(best["budgets"]["2048"]["symbol_recall_mean"], 3))
            if "c2048" in best["budgets"]:
                put("BestLearnedComposedFileRecallTwoK", round(best["budgets"]["c2048"]["file_recall_mean"], 3))
                put("BestLearnedComposedSymRecallTwoK", round(best["budgets"]["c2048"]["symbol_recall_mean"], 3))
    demo = load(R3 / "gdk_demo" / "receipts.json")
    if demo and demo.get("summary"):
        for key, name in (("packet", "Packet"), ("no_packet", "NoPacket")):
            s = demo["summary"].get(key, {})
            put(f"Gdk{name}Runs", s.get("runs"))
            put(f"Gdk{name}FileNamed", s.get("file_named"))
            put(f"Gdk{name}FunctionNamed", s.get("function_named"))
            put(f"Gdk{name}InputTokens", round(s.get("provider_input_tokens_mean", 0)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    numbers: dict = {}
    mcp02_tables(numbers)
    mcp03_tables(numbers)
    gdk_table(numbers)
    macros(numbers)
    (Path(__file__).resolve().parent / "numbers.json").write_text(json.dumps(numbers, indent=1) + "\n", encoding="utf-8")
    print("tables:", sorted(p.name for p in OUT.glob("*.tex")) if OUT.is_dir() else [])


if __name__ == "__main__":
    main()
