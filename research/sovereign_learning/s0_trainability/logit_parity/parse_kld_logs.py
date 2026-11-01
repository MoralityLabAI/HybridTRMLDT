"""Collect llama-perplexity KL-divergence statistics from run logs into one receipt.

Usage: python parse_kld_logs.py WORK_DIR RECEIPT.json [--label NAME]
Reads 01_*.log (reference pass) and each later NN_*.log comparison pass.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

PATTERNS = {
    "ppl_q": r"Mean PPL\(Q\)\s*:\s*([\d.]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([\d.]+)",
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)",
    "kld_mean": r"Mean\s+KLD:\s*([\d.]+)",
    "kld_max": r"Maximum KLD:\s*([\d.]+)",
    "kld_p999": r"99\.9%\s+KLD:\s*([\d.]+)",
    "kld_median": r"Median KLD:\s*([\d.]+)",
    "rms_dp_pct": r"RMS \S+\s*:\s*([\d.]+)",
    "same_top_p_pct": r"Same top p:\s*([\d.]+)",
}


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {k: (float(m.group(1)) if (m := re.search(p, text)) else None) for k, p in PATTERNS.items()}
    if (m := re.search(r"Final estimate: PPL = ([\d.]+)", text)):
        out["final_ppl"] = float(m.group(1))
    out["gpu_assignment_seen"] = bool(re.search(r"assigned to device CUDA", text))
    return {k: v for k, v in out.items() if v is not None}


def main() -> int:
    work, receipt = Path(sys.argv[1]), Path(sys.argv[2])
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else work.name
    logs = sorted(work.glob("[0-9][0-9]_*.log"))
    passes = {p.stem: parse(p) for p in logs}
    comparisons = {k: v for k, v in passes.items() if "same_top_p_pct" in v}
    result = {
        "schema": "s0_logit_parity.v1", "label": label,
        "passes": passes,
        "primary_criterion": "same_top_p_pct >= 99.0 per comparison",
        "primary_pass": {k: v["same_top_p_pct"] >= 99.0 for k, v in comparisons.items()},
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["primary_pass"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
