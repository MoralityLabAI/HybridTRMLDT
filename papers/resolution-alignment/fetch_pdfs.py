"""Optional arXiv PDF retrieval for the Resolution reference catalog.

Downloads are local research copies, excluded from Git. No downloads occur
unless --fetch is explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_PDF_BYTES = 40 * 1024 * 1024


def indexed_pdfs(catalog: Path = HERE / "catalog.json") -> list[tuple[str, str, str]]:
    data = json.loads(catalog.read_text(encoding="utf-8"))
    if len(data["papers"]) != 28:
        raise ValueError("expected 28 indexed Resolution entries")
    rows = []
    for item in data["papers"]:
        if not item.get("pdf_url"):
            continue
        paper_id = item["url"].removeprefix("https://arxiv.org/abs/")
        if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", paper_id):
            raise ValueError(f"noncanonical arXiv id: {paper_id!r}")
        rows.append((item["id"], paper_id, item["pdf_url"]))
    return rows


def fetch_one(url: str, target: Path, *, max_bytes: int = MAX_PDF_BYTES) -> str:
    if target.exists():
        return "already_present"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "HybridTRMLDT-ResolutionIndex/0.1 (research PDF retrieval)"},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            if int(response.headers.get("Content-Length", "0")) > max_bytes:
                raise ValueError(f"PDF too large: {url}")
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".pending-", suffix=".pdf", delete=False
            ) as temp:
                tmp_path = Path(temp.name)
                total = 0
                first = response.read(5)
                if first != b"%PDF-":
                    raise ValueError(f"response is not PDF: {url}")
                temp.write(first)
                total += len(first)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"PDF exceeded {max_bytes} bytes: {url}")
                    temp.write(chunk)
        tmp_path.replace(target)
        return "downloaded"
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="List or fetch arXiv PDFs in the Resolution index.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="Show indexed PDFs without a download (default)")
    mode.add_argument("--fetch", action="store_true", help="Download local PDFs explicitly")
    parser.add_argument("--out-dir", type=Path, default=HERE / "pdfs")
    args = parser.parse_args()
    for entry_id, arxiv_id, url in indexed_pdfs():
        target = args.out_dir / f"{entry_id}_{arxiv_id}.pdf"
        if args.fetch:
            try:
                status = fetch_one(url, target)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                status = f"ERROR: {exc}"
            print(f"{entry_id}  {status}  {target}")
        else:
            print(f"{entry_id}  {url}  -> {target}")


if __name__ == "__main__":
    main()
