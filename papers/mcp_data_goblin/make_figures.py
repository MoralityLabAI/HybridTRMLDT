"""Figures for the paper from committed results (needs matplotlib).

    python papers/mcp_data_goblin/make_figures.py

Figure 1: MCP-03 packet tokens against gold-file recall (search arm, router packets at 512-4096 tokens) for a
focused set of routers, with SWE-bench's BM25 whole-file context and BM25 path lists as reference points.
Figure 2: MCP-02 layout sweep at 1024 resources: tool-result kilobytes and accepted fraction per layout.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent / "figures"
FOCUS = ("lexical_bm25", "identifier_rules", "sham_random", "oracle_upper_bound", "looped_p4_d64", "trm_c2i1_d64",
         "transformer_l4_d64", "hrm_h2l1_d64", "dense_d64")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def figure_packets() -> None:
    s = load(ROOT / "MCP/results/mcp03/swebench_lite/summary.json")
    if not s:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for g in s["groups"]:
        if g["arm"] != "search" or g["policy"] not in FOCUS:
            continue
        pts = sorted((g["budgets"][b]["packet_tokens_mean"], g["budgets"][b]["file_recall_mean"])
                     for b in ("512", "1024", "2048", "4096") if b in g["budgets"])
        style = "--" if g["family"] in ("sham", "oracle") else "-"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], style, marker="o", ms=3, lw=1.2, label=g["policy"])
    base = sorted((v["tokens_mean"], v["file_recall_mean"]) for v in s["bm25_file_baseline"].values())
    ax.plot([b[0] for b in base], [b[1] for b in base], "k:", marker="s", ms=4, lw=1.5, label="BM25 whole files")
    maps = sorted((v["tokens_mean"], v["file_recall_mean"]) for v in s.get("bm25_path_map_baseline", {}).values())
    if maps:
        ax.plot([m[0] for m in maps], [m[1] for m in maps], "k--", marker="^", ms=4, lw=1, label="BM25 path list (5/12/25)")
    gold = s["context"]["gold_files_tokens_mean"]
    if gold:
        ax.axvline(gold, color="grey", lw=0.8)
        ax.text(gold * 1.05, 0.9, "gold files\nin full", fontsize=7, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("handoff tokens per task (Qwen2.5 tokenizer)")
    ax.set_ylabel("gold-file recall")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    ax.set_title("SWE-bench Lite, 109 tasks from 10 held-out repositories (search arm)", fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "mcp03_tokens_vs_recall.pdf")
    fig.savefig(FIG / "mcp03_tokens_vs_recall.png", dpi=160)
    plt.close(fig)


def figure_layout() -> None:
    s = load(ROOT / "MCP/results/mcp02/pilot_gpu/summary.json")
    if not s:
        return
    rows = [g for g in s["groups"] if g["experiment"] == "layout"]
    if not rows:
        return
    layouts = sorted({g["layout"] for g in rows}, key=lambda l: (len(l), l))
    xs = {l: i for i, l in enumerate(layouts)}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    policies = sorted({g["policy"] for g in rows})
    for k, policy in enumerate(policies):
        for arm, marker in (("layered", "o"), ("layered_ldt", "x")):
            sel = [g for g in rows if g["policy"] == policy and g["arm"] == arm]
            if not sel:
                continue
            off = (k - len(policies) / 2) * 0.08 + (0.03 if arm == "layered_ldt" else 0)
            color = f"C{k}"
            axes[0].scatter([xs[g["layout"]] + off for g in sel], [g["tool_result_bytes_mean"] / 1024 for g in sel],
                            marker=marker, s=18, color=color, label=f"{policy} ({arm})")
            axes[1].scatter([xs[g["layout"]] + off for g in sel], [g["accepted"] / g["episodes"] for g in sel],
                            marker=marker, s=18, color=color)
    for ax in axes:
        ax.set_xticks(range(len(layouts)))
        ax.set_xticklabels([l or "flat" for l in layouts], fontsize=7)
        ax.set_xlabel("layout (facets, root to leaf)")
    axes[0].set_ylabel("tool-result KB per episode")
    axes[1].set_ylabel("accepted fraction")
    axes[0].legend(fontsize=5.5, loc="upper right")
    fig.suptitle("MCP-02 layout sweep at 1024 resources (o layered, x certified LDT)", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "mcp02_layout_depth.pdf")
    fig.savefig(FIG / "mcp02_layout_depth.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    figure_packets()
    figure_layout()
    print("figures:", sorted(p.name for p in FIG.glob("*")) if FIG.is_dir() else [])
