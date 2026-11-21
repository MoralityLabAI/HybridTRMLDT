"""Tests for the hub-run packet compaction and relay (stdlib only)."""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request

from MCP.hubrun.packet import PACKET_HEADER, compact_messages, estimate_tokens
from MCP.hubrun.proxy import Relay, make_handler, normalize_messages


def _episode(steps: int, noise_lines: int = 60) -> list[dict]:
    msgs = [{"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Find the invoice INV-4471 total and reconcile ledger_east against ledger_west."}]
    for i in range(steps):
        call_id = f"c{i}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "query", "arguments": json.dumps({"q": f"page {i}"})}}]})
        body = [f"row {i}-{j}: unrelated filler value {j * 7}" for j in range(noise_lines)]
        if i == 1:
            body.insert(10, "INV-4471 total 9,812.40 recorded in ledger_east only")
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": "\n".join(body)})
    return msgs


class PacketTests(unittest.TestCase):
    def test_below_threshold_is_unchanged(self):
        msgs = _episode(2, noise_lines=3)
        out, receipt = compact_messages(msgs, threshold_tokens=10_000)
        self.assertEqual(out, msgs)
        self.assertFalse(receipt["compacted"])

    def test_compaction_keeps_task_tail_ledger_and_exact_evidence(self):
        msgs = _episode(8)
        out, receipt = compact_messages(msgs, threshold_tokens=1000, budget_tokens=600, keep_recent=2)
        self.assertTrue(receipt["compacted"])
        self.assertEqual(out[:2], msgs[:2])
        self.assertEqual(out[-4:], msgs[-4:])
        packet = out[2]["content"]
        self.assertTrue(packet.startswith(PACKET_HEADER))
        self.assertIn("INV-4471 total 9,812.40", packet)
        for i in range(6):
            self.assertIn(f"step {i + 1}: query(", packet)
        self.assertLess(estimate_tokens(out), estimate_tokens(msgs) / 2)
        self.assertEqual(receipt["ledger_calls"], 6)

    def test_every_tool_message_follows_its_call(self):
        out, _ = compact_messages(_episode(9), threshold_tokens=1000, budget_tokens=500, keep_recent=3)
        open_calls: set[str] = set()
        for m in out:
            if m["role"] == "assistant":
                open_calls = {c["id"] for c in m.get("tool_calls") or []}
            if m["role"] == "tool":
                self.assertIn(m["tool_call_id"], open_calls)

    def test_deterministic(self):
        a = compact_messages(_episode(7), threshold_tokens=800)
        b = compact_messages(_episode(7), threshold_tokens=800)
        self.assertEqual(a, b)

    def test_long_tail_result_is_capped(self):
        msgs = _episode(1, noise_lines=4000)
        out, receipt = compact_messages(msgs, threshold_tokens=10**9, per_message_cap=500)
        self.assertEqual(receipt["capped_messages"], 1)
        self.assertLess(estimate_tokens(out[-1]["content"]), 600)


class NormalizeTests(unittest.TestCase):
    def test_nulls_and_object_arguments(self):
        msgs = [{"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]},
                {"role": "tool", "tool_call_id": "a", "content": "ok", "name": None},
                {"role": "user", "content": "go", "tool_calls": None}]
        out = normalize_messages(msgs)
        self.assertEqual(out[0]["content"], "")
        self.assertNotIn("reasoning_content", out[0])
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], '{"x": 1}')
        self.assertNotIn("name", out[1])
        self.assertNotIn("tool_calls", out[2])
        self.assertIsNone(msgs[0]["content"])


class _Upstream(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append(body)
        payload = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 11, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        return


class RelayTests(unittest.TestCase):
    def test_goblin_arm_compacts_and_logs_usage(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            relay = Relay(f"http://127.0.0.1:{upstream.server_port}", log, None, 512, 1000, 600, 2, 6000, 30)
            proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(relay))
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            for arm in ("raw", "goblin"):
                req = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/{arm}/t1/v1/chat/completions",
                                             data=json.dumps({"model": "m", "messages": _episode(8)}).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self.assertEqual(json.loads(resp.read())["choices"][0]["message"]["content"], "ok")
            proxy.shutdown()
            rows = [json.loads(line) for line in log.read_text().splitlines()]
        upstream.shutdown()
        self.assertEqual([r["arm"] for r in rows], ["raw", "goblin"])
        self.assertEqual(rows[0]["prompt_tokens"], 11)
        self.assertFalse(rows[0]["compaction"]["compacted"])
        self.assertTrue(rows[1]["compaction"]["compacted"])
        self.assertEqual(_Upstream.seen[0]["max_tokens"], 512)
        self.assertLess(len(_Upstream.seen[1]["messages"]), len(_Upstream.seen[0]["messages"]))


class SummarizeTests(unittest.TestCase):
    def _write(self, root: Path) -> Path:
        runs = root / "runs"
        v1 = runs / "er-t1-raw" / "x"
        v1.mkdir(parents=True)
        (runs / "er-t1-raw" / "DONE").write_text("t")
        trace = lambda name, s: {"task": {"data": {"name": name}}, "errors": [], "traces": [{
            "ok": True, "stop_condition": "agent_completed", "errors": [],
            "rewards": {"a": {"score": s, "weight": 0.5}, "b": {"score": 1.0, "weight": 0.5}},
            "metrics": {"turns": 4.0}}]}
        (v1 / "traces.jsonl").write_text("\n".join(json.dumps(trace(n, s)) for n, s in [("t1", 1.0), ("t2", 0.0)]))
        gob = runs / "er-t1-goblin" / "x"
        gob.mkdir(parents=True)
        (gob / "traces.jsonl").write_text("\n".join(json.dumps(trace(n, s)) for n, s in [("t1", 1.0), ("t2", 1.0)]))
        legacy = runs / "sw-constitution-eval" / "outputs"
        legacy.mkdir(parents=True)
        rows = [{"example_id": 0, "reward": 0.0, "answer": "opt_a", "completion": [{"role": "assistant", "content": "<opt_a>"}],
                 "num_turns": 1}, {"example_id": 1, "reward": 1.0, "answer": "opt_b",
                                   "completion": [{"role": "assistant", "content": "opt_b"}], "num_turns": 1}]
        (legacy / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        sq = runs / "sqlite-all" / "x"
        sq.mkdir(parents=True)
        sq_trace = {"task": {"data": {"name": "q1"}}, "errors": [], "traces": [{
            "ok": True, "errors": [], "rewards": {"tables": {"score": 1.0, "weight": 0.0}}, "metrics": {"exact_match": 1.0}}]}
        (sq / "traces.jsonl").write_text(json.dumps(sq_trace))
        calls = root / "calls.jsonl"
        calls.write_text("\n".join(json.dumps(r) for r in [
            {"tag": "er-t1-raw", "arm": "raw", "status": 200, "prompt_tokens": 30000, "completion_tokens": 10,
             "finish_reason": "length", "compaction": {"compacted": False}},
            {"tag": "er-t1-raw", "arm": "raw", "status": 400, "compaction": {"compacted": False},
             "error": '{"error": {"type": "exceed_context_size_error"}}'},
            {"tag": "er-t1-goblin", "arm": "goblin", "status": 200, "prompt_tokens": 9000, "completion_tokens": 10,
             "finish_reason": "stop", "compaction": {"compacted": True}}]))
        return calls

    def test_scores_pairs_privacy_and_primary_metric(self):
        from MCP.hubrun.summarize import load_calls, load_episodes, pairs, summarize_run

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = load_calls(self._write(root))
            runs, eps = {}, {}
            for tag in ("er-t1-raw", "er-t1-goblin", "sw-constitution-eval", "sqlite-all"):
                eps[tag] = load_episodes(root / "runs" / tag, tag)
                runs[tag] = summarize_run(tag, eps[tag], calls.get(tag, {}))
        self.assertEqual(runs["er-t1-raw"]["mean_score"], 0.75)
        self.assertEqual(runs["er-t1-raw"]["solved"], 1)
        self.assertEqual(runs["sqlite-all"]["mean_score"], 1.0)
        self.assertNotIn("per_task", runs["sw-constitution-eval"])
        self.assertIn("per_task", runs["er-t1-raw"])
        self.assertEqual(runs["sw-constitution-eval"]["metric_means"]["bracket_stripped_choice_match"], 1.0)
        pair = pairs(runs, eps)["er-t1"]
        self.assertEqual((pair["goblin_better"], pair["raw_better"]), (1, 0))
        self.assertEqual(pair["prompt_tokens_ratio_goblin_over_raw"], 0.3)
        self.assertEqual(pair["context_length_stops_raw"], 1)
        self.assertEqual(pair["context_overflows_raw"], 1)
        self.assertEqual(pair["context_overflows_goblin"], 0)


if __name__ == "__main__":
    unittest.main()
