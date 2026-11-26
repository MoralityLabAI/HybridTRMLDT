"""Receipt Goblin: number semantics, claim extraction, receipt leaves, packets, server flow, benchmark integrity."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from MCP.receipt_goblin.bench import HERE, ROOT, Indexes, item_parts, load_bench
from MCP.receipt_goblin.claims import extract
from MCP.receipt_goblin.numbers import matches, parse, parse_one, render
from MCP.receipt_goblin.packet import build, certify, number_matches
from MCP.receipt_goblin.receipts import ReceiptIndex
from MCP.receipt_goblin.server import ReceiptService

FIXTURE = {"runs": [
    {"arm": "raw", "policy": "bm25", "catalog_size": 256, "accepted": 12, "tool_calls_mean": 31.5},
    {"arm": "goblin", "policy": "looped", "catalog_size": 256, "accepted": 27, "tool_calls_mean": 6.25},
], "episodes": 78512}
DOC = ("# Results\n\nAccepted tasks per arm out of 32, at catalog size 256.\n\n"
       "| Arm | Accepted | Tool calls |\n| --- | --- | --- |\n| raw | 12 | 31.5 |\n| goblin | 12 | 6.25 |\n\n"
       "The run used 78.5K episodes, 0.90–0.92 accuracy and `MCP-02` on Qwen2.5-0.5B in 2026.\n")


def fixture_dir() -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "receipts").mkdir()
    (root / "receipts" / "summary.json").write_text(json.dumps(FIXTURE), encoding="utf-8")
    (root / "doc.md").write_text(DOC, encoding="utf-8")
    return tmp


class NumberTests(unittest.TestCase):
    def test_parse_skips_identifiers_versions_and_dates(self):
        surfaces = [n.surface for n in parse("Qwen2.5-0.5B MCP-02 swebench 5.0.2 on 2026-11-20 d128 used 13,430 and 0.86")]
        self.assertEqual(surfaces, ["13,430", "0.86"])

    def test_written_precision_and_scale(self):
        self.assertTrue(matches(parse_one("0.86"), 0.8612))
        self.assertFalse(matches(parse_one("0.86"), 0.8549))
        self.assertTrue(matches(parse_one("78.5K"), 78512))
        self.assertTrue(matches(parse_one("15%"), 0.15))
        self.assertTrue(matches(parse_one("32"), 32.0))
        self.assertFalse(matches(parse_one("32"), 31.6 + 1.0))
        self.assertFalse(matches(parse_one("1"), True))

    def test_render_keeps_style(self):
        self.assertEqual(render(13430, parse_one("12,000")), "13,430")
        self.assertEqual(render(0.8571, parse_one("0.86")), "0.86")
        self.assertEqual(render(0.15, parse_one("80%")), "15%")
        self.assertEqual(render(78512, parse_one("1.0K")), "78.5K")


class ClaimAndIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = fixture_dir()
        self.root = Path(self.tmp.name)
        self.index = ReceiptIndex(self.root / "receipts")
        self.claims = extract(self.root / "doc.md", "doc.md")

    def tearDown(self):
        self.tmp.cleanup()

    def test_table_rows_carry_their_lead_but_not_its_numbers(self):
        rows = [c for c in self.claims if c.line == 8]
        self.assertEqual([c.target.surface for c in rows], ["12", "6.25"])
        self.assertTrue(rows[0].text.startswith("Accepted tasks per arm out of 32, at catalog size 256. | Arm: goblin"))
        self.assertEqual(rows[0].cid, "doc.md:8:23")

    def test_ranges_years_and_code_are_not_claims(self):
        prose = [c.target.surface for c in self.claims if c.line == 10]
        self.assertEqual(prose, ["78.5K"])

    def test_list_elements_are_labelled_and_label_echoes_dropped(self):
        paths = {leaf.path for leaf in self.index.leaves}
        self.assertIn("summary.json :: runs/[arm=goblin policy=looped catalog_size=256]/accepted", paths)
        self.assertFalse(any(path.endswith("/catalog_size") for path in paths))

    def test_siblings_differ_in_one_component(self):
        lid = next(l.lid for l in self.index.leaves if l.path.endswith("[arm=goblin policy=looped catalog_size=256]/accepted"))
        names = {self.index.leaves[o].path for o, _ in self.index.siblings(lid)}
        self.assertIn("summary.json :: runs/[arm=raw policy=bm25 catalog_size=256]/accepted", names)
        self.assertIn("summary.json :: runs/[arm=goblin policy=looped catalog_size=256]/tool_calls_mean", names)

    def test_packet_and_certify(self):
        claim = next(c for c in self.claims if c.line == 8 and c.target.surface == "12")
        packet = build(self.index, claim.context(), claim.target, k_neighbours=8)
        paths = {self.index.leaves[lid].path: key for key, lid in packet.ids.items()}
        goblin = paths["summary.json :: runs/[arm=goblin policy=looped catalog_size=256]/accepted"]
        raw = paths["summary.json :: runs/[arm=raw policy=bm25 catalog_size=256]/accepted"]
        self.assertEqual(certify(self.index, claim.target, packet.ids[goblin])["verdict"], "contradicted")
        self.assertEqual(certify(self.index, claim.target, packet.ids[raw])["verdict"], "supported")
        self.assertEqual(certify(self.index, claim.target, None)["verdict"], "unsupported")
        self.assertEqual(len(number_matches(self.index, claim.target)), 1)
        blind = build(self.index, claim.context(), claim.target, k_neighbours=8, blind=True)
        self.assertNotIn(" = ", blind.text)
        self.assertEqual(set(blind.ids.values()), set(packet.ids.values()))


class ServerTests(unittest.TestCase):
    def test_extract_packet_certify_flow(self):
        with fixture_dir() as tmp:
            root = Path(tmp)
            service = ReceiptService()
            claims = service.call("extract_claims", {"doc_path": str(root / "doc.md")})["structuredContent"]["claims"]
            goblin = next(c for c in claims if c["target"] == "12" and "goblin" in c["text"])
            packet = service.call("claim_packet", {"text": goblin["text"], "start": goblin["start"], "end": goblin["end"],
                                                   "receipts_root": str(root / "receipts")})
            self.assertFalse(packet["isError"])
            body = packet["structuredContent"]
            self.assertTrue(packet["content"][0]["text"].startswith(f"packet_id: {body['packet_id']}"))
            key = next(line.split()[0] for line in body["packet"].splitlines()
                       if "arm=goblin" in line and line.split(" = ")[0].endswith("/accepted"))
            verdict = service.call("certify", {"packet_id": body["packet_id"], "leaf": key})["structuredContent"]
            self.assertEqual(verdict["verdict"], "contradicted")
            self.assertTrue(service.call("certify", {"packet_id": body["packet_id"], "leaf": "L99"})["isError"])
            by_target = service.call("claim_packet", {"text": goblin["text"], "target": "6.25",
                                                      "receipts_root": str(root / "receipts")})
            self.assertEqual(by_target["structuredContent"]["target"], "6.25")


class BenchIntegrityTests(unittest.TestCase):
    """The committed benchmark still agrees with the committed receipts."""

    def test_items_resolve_against_receipts(self):
        indexes = Indexes()
        items = load_bench()
        self.assertGreaterEqual(len(items), 100)
        for item in items:
            index = indexes[item["receipts"]]
            by_path = {leaf.path: leaf for leaf in index.leaves}
            target, _ = item_parts(item)
            self.assertEqual(item["text"][item["start"]:item["end"]], item["target"])
            gold = [by_path[p] for p in item["gold_leaves"]]
            if item["variant"] == "clean":
                self.assertTrue(all(matches(target, leaf.value) for leaf in gold), item["id"])
            else:
                self.assertFalse(any(matches(target, leaf.value) for leaf in gold), item["id"])
            if item["variant"] == "decoy":
                self.assertTrue(matches(target, by_path[item["decoy_leaf"]].value), item["id"])
            if item["variant"] == "off_receipt":
                self.assertEqual(number_matches(index, target), [], item["id"])

    def test_gold_file_is_the_source(self):
        gold = json.loads((HERE / "gold.json").read_text(encoding="utf-8"))["claims"]
        clean = {item["cid"] for item in load_bench() if item["variant"] == "clean"}
        self.assertEqual(clean, {cid for cid, entry in gold.items() if entry["leaves"]})
        self.assertTrue((ROOT / "MCP/receipt_goblin/bench_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
