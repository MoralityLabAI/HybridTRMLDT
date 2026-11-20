"""Figures for the paper from committed results (needs matplotlib).

    python papers/mcp_data_goblin/make_figures.py

Figure 1: MCP-03 packet tokens against gold-file recall for every router and budget, with the
SWE-bench BM25 whole-file baseline and the mean size of the gold files in full.
Figure 2: MCP-02 index depth against tool-result kilobytes and accepted episodes at 1024 resources.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent / "figures"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def figure_packets() -> None:
    s = load(ROOT / "MCP/results/mcp03/swebench_lite/summary.json")
    if not s:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for g in s["groups"]:
        if g["arm"] != "hybrid" and g["arm"] != "search":
            continue
        pts = sorted((v["packet_tokens_mean"], v["file_recall_mean"]) for v in g["budgets"].values())
        style = "--" if g["family"] in ("sham", "oracle") else "-"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], style, marker="o", ms=3, lw=1,
                label=f"{g['policy']} ({g['arm']})")
    base = sorted((v["tokens_mean"], v["file_recall_mean"]) for v in s["bm25_file_baseline"].values())
    ax.plot([b[0] for b in base], [b[1] for b in base], "k:", marker="s", ms=4, lw=1.5, label="BM25 whole files")
    gold = s["context"]["gold_files_tokens_mean"]
    if gold:
        ax.axvline(gold, color="grey", lw=0.8)
        ax.text(gold, 0.02, " gold files in full", fontsize=7, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("handoff tokens (Qwen2.5 tokenizer, mean per task)")
    ax.set_ylabel("gold-file recall")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.set_title("SWE-bench Lite, 109 tasks from 10 held-out repositories", fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "mcp03_tokens_vs_recall.pdf")
    fig.savefig(FIG / "mcp03_tokens_vs_recall.png", dpi=160)


def figure_layout() -> None:
    for run in ("pilot_gpu", "pilot_cpu"):
        s = load(ROOT / f"MCP/results/mcp02/{run}/summary.json")
        if s and any(g["experiment"] == "layout" for g in s["groups"]):
            break
    else:
        return
    rows = [g for g in s["groups"] if g["experiment"] == "layout"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for policy in sorted({g["policy"] for g in rows}):
        for arm, ls in (("layered", "-"), ("layered_ldt", "--")):
            sel = sorted((len(g["layout"]) + 1, g["tool_result_bytes_mean"] / 1024, g["accepted"] / g["episodes"])
                         for g in rows if g["policy"] == policy and g["arm"] == arm)
            if not sel:
                continue
            axes[0].plot([x[0] for x in sel], [x[1] for x in sel], ls, marker="o", ms=3, label=f"{policy} {arm}")
            axes[1].plot([x[0] for x in sel], [x[2] for x in sel], ls, marker="o", ms=3)
    axes[0].set_xlabel("index depth (facets + leaf)")
    axes[0].set_ylabel("tool-result KB per episode")
    axes[1].set_xlabel("index depth (facets + leaf)")
    axes[1].set_ylabel("accepted fraction")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"MCP-02 layout sweep at 1024 resources ({run})", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "mcp02_layout_depth.pdf")
    fig.savefig(FIG / "mcp02_layout_depth.png", dpi=160)


if __name__ == "__main__":
    figure_packets()
    figure_layout()
    print("figures:", sorted(p.name for p in FIG.glob("*")) if FIG.is_dir() else [])
