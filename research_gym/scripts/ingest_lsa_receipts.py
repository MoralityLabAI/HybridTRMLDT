"""Normalize LSA receipts into censor-aware LSPG evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from research_gym.prognostics.empirical import ingest_lsa_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = asdict(ingest_lsa_receipt(args.receipt))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
