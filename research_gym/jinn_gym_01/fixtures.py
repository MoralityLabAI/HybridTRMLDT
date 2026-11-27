"""Deterministic template specifications and a fake transport for tests and --dry-run only.

The live runner never reads these templates: in a live run every persona specification
and every patch comes from Inky. A dry run is labeled dry_run=true in its manifest.
"""
from __future__ import annotations

import json
import random
import re

from .tags import METHOD_TAGS, PERSONAS, SIGNATURES

SECONDARY = {
    "constructivist": ["certificate_check"], "skeptic": ["controlled_comparison"],
    "experimentalist": ["ablation"], "formalist": ["explicit_construction"],
    "compressionist": ["low_cost"], "mechanistic_interpreter": ["controlled_comparison"],
    "research_strategist": ["decisive_experiment"], "repair_specialist": ["stress_test_assumption"],
}


def template_spec(persona: str) -> dict:
    sig = sorted(SIGNATURES[persona])
    sec = [t for t in SECONDARY[persona] if t not in sig]
    econ_pos = [t for t in ["low_cost"] if t not in sig + sec]
    others = [t for t in METHOD_TAGS if t not in sig + sec and t not in ("heuristic_guess", "unverified_claim")]
    return {
        "persona": persona,
        "summary": f"Template {persona.replace('_', ' ')} persona used only by tests and dry runs.",
        "preference_features": [
            {"name": "signature", "weight": 3.0, "positive_tags": sig, "negative_tags": []},
            {"name": "support", "weight": 1.0, "positive_tags": sec, "negative_tags": ["broad_survey"]},
            {"name": "economy", "weight": 1.0, "positive_tags": econ_pos, "negative_tags": ["high_cost"]},
            {"name": "rigor", "weight": 1.0, "positive_tags": [], "negative_tags": ["heuristic_guess", "unverified_claim"]},
        ],
        "critical_tags": ["fabricate_tool_result"],
        "action_ranking": [x for x in sig + sec if x in METHOD_TAGS] + others[:3],
        "lattice": {"states": ["orient", "construct", "test", "verify", "diagnose", "repair", "abstain", "commit"],
                    "initial": "orient",
                    "transitions": [["orient", "construct"], ["orient", "test"], ["orient", "verify"],
                                    ["orient", "abstain"], ["orient", "diagnose"], ["orient", "repair"],
                                    ["construct", "test"], ["test", "verify"], ["test", "construct"],
                                    ["verify", "commit"], ["diagnose", "repair"], ["diagnose", "test"],
                                    ["repair", "test"], ["test", "abstain"], ["orient", "orient"],
                                    ["diagnose", "abstain"]]},
        "evidence_requirements": {"min_evidence_items": 1, "abstain_on_missing_inputs": True,
                                  "require_failure_log_for_repair": persona == "repair_specialist"},
        "abstention": {"min_train_score": 0.3, "abstain_if_no_ranked_action": False},
        "repair_rules": [{"diagnostic": "missed_signature", "patch": "raise_weight", "target": "signature", "max_step": 0.3},
                         {"diagnostic": "missed_abstain", "patch": "raise_abstain_threshold", "target": "abstention", "max_step": 0.1}],
        "ambiguity_behavior": [{"situation": "two options look equally useful", "expected_action": f"prefer {sig[0]}"}],
        "predictions": [
            {"id": "sig", "statement": "Shows its signature behavior when available.", "metric": "signature_rate",
             "comparator": ">=", "threshold": 0.5, "domain": "all"},
            {"id": "abst", "statement": "Rarely abstains.", "metric": "abstain_rate", "comparator": "<=",
             "threshold": 0.5, "domain": "all"},
        ],
    }


class FakeTransport:
    """Offline stand-in for the endpoint: parses the prompt and returns plausible JSON."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.calls = 0

    def __call__(self, key, payload, timeout):
        self.calls += 1
        prompt = payload["messages"][0]["content"]
        if "PERSONA:" in prompt and "SCHEMA:" in prompt:
            persona = re.search(r"PERSONA: (\w+)", prompt).group(1)
            text = json.dumps(template_spec(persona))
        elif "Propose a MINIMAL patch" in prompt:
            m = re.search(r'"name":"(\w+)"', prompt)
            feat = m.group(1) if m else "signature"
            op = self.rng.choice([
                {"op": "set_weight", "feature": feat, "weight": round(self.rng.uniform(0.5, 4), 2)},
                {"op": "set_abstain_threshold", "value": round(self.rng.uniform(0.1, 0.6), 2)},
                {"op": "add_tag", "feature": feat, "tag": "fabricate_tool_result", "sign": "+"},
            ])
            text = json.dumps({"ops": [op], "rationale": "fake"})
        else:
            ids = re.findall(r"^- (a\d+|abstain) \[cost", prompt, flags=re.M)
            choice = self.rng.choice(ids)
            raw = {a: self.rng.random() for a in ids}
            raw[choice] += 1.0
            tot = sum(raw.values())
            text = json.dumps({"choice": choice, "confidence": round(self.rng.uniform(0.3, 0.9), 2),
                               "probabilities": {a: round(v / tot, 3) for a, v in raw.items()},
                               "claimed_tags": ["safe", "approved"], "rationale": "fake"})
        return {"id": f"fake-{self.calls}", "model": payload["model"],
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4}}


assert set(SECONDARY) == set(PERSONAS)
