import json
from pathlib import Path

from research_gym.scripts.export_receipt_sft import SCHEMA_VERSION
import runpy


def test_resolution_catalog_count_urls_and_no_duplicates():
    path = Path("papers/resolution-alignment/catalog.json")
    catalog = json.loads(path.read_text(encoding="utf-8"))
    assert len(catalog["papers"]) == 28
    assert len({row["id"] for row in catalog["papers"]}) == 28
    assert len({row["url"] for row in catalog["papers"]}) == 28
    assert all(row["url"].startswith("https://") for row in catalog["papers"])
    assert all(
        row["pdf_url"] == row["url"].replace("/abs/", "/pdf/")
        for row in catalog["papers"]
        if row["pdf_url"]
    )
    assert len(catalog["additional_timaeus_posts"]) == 1


def test_silico_config_points_to_new_exporter():
    cfg = json.loads(Path("configs/silico_rivet_v1.json").read_text(encoding="utf-8"))
    assert cfg["registration_status"] == "proposed_unlaunched"
    assert cfg["preflight"]["leakage_policy"] == "input_allowlist_only"
    assert "export_receipt_sft" in cfg["dataset"]["copyable_command"]
    assert cfg["resources"]["paid_compute_auto_launch"] is False
    assert cfg["promotion_gates"]["zero_unauthorized_hard_updates"] is True
    assert SCHEMA_VERSION == "rivet_stage1_sft_v1"
