"""Checks for source integrity, exact metadata filtering and public replay."""
import copy
import json
import unittest

from MCP.environment import (
    ARCHITECTURES, RetrievalEnv, choose_reference_action, make_cases,
)


def run_reference(case, architecture, *, depth=3, catalog_size=32):
    env = RetrievalEnv(case, architecture, depth=depth, catalog_size=catalog_size)
    history = []
    for _ in range(200):
        action = choose_reference_action(env.task, env.tools(), history)
        if action["tool"] == "finish":
            return env, action["arguments"], history
        result = env.call(action["tool"], action["arguments"])
        if result["isError"]:
            raise AssertionError(result)
        history.append({**action, "result": result})
    raise AssertionError("reference controller exceeded its tool-call cap")


class FixtureTests(unittest.TestCase):
    def test_deterministic_fixtures_and_depth_independent_gold(self):
        first = make_cases(n=10, depth=1)
        self.assertEqual(first, make_cases(n=10, depth=5))
        self.assertEqual({case["family"] for case in first},
                         {"lookup", "difference", "cross_branch_sum", "stale_version", "missing_evidence"})
        self.assertTrue(all(case["provenance"]["kind"] == "synthetic" for case in first))

    def test_split_groups_and_source_objects_are_disjoint(self):
        cases = {split: make_cases(split=split, seed=11, n=8) for split in ("train", "dev", "test")}
        sources = {split: {row["ref"] for case in rows for row in case["corpus"]}
                   for split, rows in cases.items()}
        groups = {split: {case["group_id"] for case in rows} for split, rows in cases.items()}
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            self.assertTrue(sources[left].isdisjoint(sources[right]))
            self.assertTrue(groups[left].isdisjoint(groups[right]))

    def test_catalog_widths_share_task_gold_and_nested_source_objects(self):
        narrow = make_cases(n=10, catalog_size=8)
        wide = make_cases(n=10, catalog_size=96)
        for first, second in zip(narrow, wide):
            self.assertEqual(first["task"], second["task"])
            self.assertEqual(first["oracle"], second["oracle"])
            wider_rows = {row["record_id"]: row for row in second["corpus"]}
            self.assertTrue(all(wider_rows[row["record_id"]] == row for row in first["corpus"]))

    def test_invalid_generator_parameters(self):
        for arguments in ({"split": "heldout"}, {"catalog_size": 7}, {"depth": 0},
                          {"depth": 13}, {"n": -1}):
            with self.assertRaises(ValueError):
                make_cases(**arguments)

    def test_public_surfaces_and_retrieval_do_not_use_the_oracle(self):
        case = make_cases(n=1)[0]
        poisoned = copy.deepcopy(case)
        poisoned["oracle"] = {"answer": 987654321, "required_refs": [], "abstain": True}
        for architecture in ARCHITECTURES:
            env = RetrievalEnv(case, architecture, depth=1)
            other = RetrievalEnv(poisoned, architecture, depth=1)
            self.assertEqual(env.task, other.task)
            self.assertEqual(env.tools(), other.tools())
            public = json.dumps({"task": env.task, "tools": env.tools()})
            for private_key in ('"oracle"', '"required_refs"', '"family"', '"expected_answer"'):
                self.assertNotIn(private_key, public)
            action = choose_reference_action(env.task, env.tools(), [])
            self.assertEqual(env.call(action["tool"], action["arguments"]),
                             other.call(action["tool"], action["arguments"]))


class RetrievalTests(unittest.TestCase):
    def test_reference_solves_every_family_in_every_arm(self):
        for case in make_cases(n=10):
            for architecture in ARCHITECTURES:
                with self.subTest(case=case["family"], architecture=architecture):
                    env, submission, history = run_reference(case, architecture)
                    self.assertTrue(env.grade(submission)["success"], env.grade(submission))
                    self.assertEqual(env.stats["tool_calls"], len(history))
                    self.assertEqual(len(env.handoff(submission)["evidence"]),
                                     len(submission["evidence_refs"]))

    def test_pagination_and_configurable_tree_depth(self):
        case = make_cases(n=3, catalog_size=96)[2]
        for architecture in ARCHITECTURES:
            for depth in (1, 5):
                env, submission, _ = run_reference(case, architecture, depth=depth, catalog_size=96)
                self.assertTrue(env.grade(submission)["success"])

    def test_discovery_has_no_values_or_citation_handles(self):
        case = make_cases(n=1)[0]
        for architecture, name, arguments in (
            ("search", "catalog_search", {"query": ""}),
            ("layered", "index_children", {"limit": 64}),
            ("layered_ldt", "index_children", {"limit": 64}),
        ):
            env = RetrievalEnv(case, architecture, depth=1)
            result = env.call(name, arguments)["structuredContent"]
            for row in result.get("records", []) + result.get("children", []):
                self.assertNotIn("value", row)
                self.assertNotIn("value", row["metadata"])
                self.assertNotIn("ref", row)

    def test_exact_ldt_removes_provenance_traps_and_keeps_missing(self):
        case = make_cases(n=5)[4]
        raw = RetrievalEnv(case, "layered", depth=1)
        checked = RetrievalEnv(case, "layered_ldt", depth=1)
        raw_rows = raw.call("index_children", {"limit": 64})["structuredContent"]["children"]
        payload = checked.call("index_children", {"limit": 64})["structuredContent"]
        self.assertEqual(len(raw_rows), 32)
        self.assertEqual(len(payload["children"]), 29)
        self.assertEqual(payload["verification"]["excluded"], 3)
        self.assertEqual(checked.stats["ldt_checks"], 1)
        missing = next(row for row in case["corpus"] if row["value"] is None)
        self.assertIn(missing["record_id"], {row["record_id"] for row in payload["children"]})

    def test_unknown_metadata_remains_a_candidate(self):
        case = make_cases(n=1)[0]
        stale = next(row for row in case["corpus"] if row["revision"] == "v1")
        stale["registry_metadata"]["revision"] = None
        env = RetrievalEnv(case, "layered_ldt", depth=1)
        payload = env.call("index_children", {"limit": 64})["structuredContent"]
        self.assertIn(stale["record_id"], {row["record_id"] for row in payload["children"]})
        self.assertEqual(payload["verification"]["excluded"], 2)

    def test_sibling_branches_remain_reachable(self):
        env = RetrievalEnv(make_cases(n=3)[2], "layered_ldt", depth=3)
        first = env.call("index_children", {})["structuredContent"]
        self.assertEqual(len(first["children"]), 2)
        left, right = [row["node_id"] for row in first["children"]]
        self.assertFalse(env.call("index_children", {"node_id": left})["isError"])
        self.assertFalse(env.call("index_children", {"node_id": right})["isError"])
        self.assertEqual(first, env.call("index_children", {})["structuredContent"])

    def test_unknown_and_malformed_calls_are_counted(self):
        env = RetrievalEnv(make_cases(n=1)[0], "search")
        attempts = (("missing_tool", {}), ("catalog_search", {}),
                    ("catalog_search", {"query": "", "limit": True}),
                    ("evidence_read", {"record_id": "missing"}),
                    ("catalog_search", {"query": "", "oracle": True}))
        for name, arguments in attempts:
            self.assertTrue(env.call(name, arguments)["isError"])
        self.assertEqual(env.stats["tool_calls"], len(attempts))


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.case = make_cases(n=1)[0]
        self.env = RetrievalEnv(self.case, "search")
        self.submission = {"answer": self.case["oracle"]["answer"], "abstain": False,
                           "evidence_refs": self.case["oracle"]["required_refs"]}
        self.target = next(row for row in self.case["corpus"]
                           if row["ref"] == self.submission["evidence_refs"][0])

    def test_exact_answer_and_unread_citation_fails(self):
        grade = self.env.grade(self.submission)
        self.assertTrue(grade["answer_correct"])
        self.assertFalse(grade["evidence_valid"])
        self.assertFalse(grade["success"])
        self.assertEqual(self.env.handoff(self.submission)["evidence"], [])

    def test_metadata_projection_does_not_certify_unseen_value(self):
        self.env.call("evidence_read", {"record_id": self.target["record_id"], "fields": []})
        self.assertFalse(self.env.grade(self.submission)["success"])
        self.env.call("evidence_read", {"record_id": self.target["record_id"], "fields": ["value"]})
        self.assertTrue(self.env.grade(self.submission)["success"])

    def test_stale_foreign_or_forged_citation_fails(self):
        for changed in ("revision", "scope", "epoch"):
            wrong = next(row for row in self.case["corpus"] if row[changed] != self.target[changed])
            self.env.call("evidence_read", {"record_id": wrong["record_id"]})
            submitted = dict(self.submission, evidence_refs=[wrong["ref"]])
            self.assertFalse(self.env.grade(submitted)["success"])
        self.assertFalse(self.env.grade(dict(self.submission, evidence_refs=["invented-ref"]))["success"])

    def test_missing_evidence_requires_a_read_and_explicit_abstention(self):
        case = make_cases(n=5)[4]
        env = RetrievalEnv(case, "search")
        guessed = {"answer": None, "abstain": True, "evidence_refs": []}
        self.assertFalse(env.grade(guessed)["success"])
        env, submission, _ = run_reference(case, "search")
        self.assertTrue(env.grade(submission)["success"])
        self.assertFalse(env.grade(dict(submission, abstain=False))["success"])
        self.assertFalse(env.grade(dict(submission, answer=0))["success"])

    def test_malformed_duplicate_or_boolean_answer_fails(self):
        self.env.call("evidence_read", {"record_id": self.target["record_id"]})
        for submission in ({}, [], dict(self.submission, answer=True),
                           dict(self.submission, evidence_refs=self.submission["evidence_refs"] * 2),
                           dict(self.submission, extra="oracle")):
            self.assertFalse(self.env.grade(submission)["success"])

    def test_hostile_terminal_values_cannot_crash_judge_or_handoff(self):
        for submission in ([], {"evidence_refs": [{}]}, dict(self.submission, answer=10 ** 400),
                           dict(self.submission, answer=float("inf")),
                           dict(self.submission, evidence_refs=[{}])):
            self.assertFalse(self.env.grade(submission)["success"])
            self.env.handoff(submission)


if __name__ == "__main__":
    unittest.main()
