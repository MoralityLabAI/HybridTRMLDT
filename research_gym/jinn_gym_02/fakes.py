"""Offline fake transport for tests and --dry-run only. Results produced with it are not evidence."""
from __future__ import annotations

import json
import random
import re

from research_gym.jinn_gym_01.tags import PERSONAS


class FakeTransport02:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.calls = 0

    def __call__(self, key, payload, timeout):
        self.calls += 1
        prompt = payload["messages"][0]["content"]
        if "Propose a MINIMAL patch" in prompt:
            names = re.findall(r'"name":"(\w+)"', prompt)
            op = self.rng.choice([
                {"op": "set_weight", "feature": self.rng.choice(names or ["x"]), "weight": round(self.rng.uniform(0.5, 6), 2)},
                {"op": "set_abstain_threshold", "value": round(self.rng.uniform(0.0, 0.6), 2)},
                {"op": "set_abstain_if_no_ranked_action", "value": False},
            ])
            text = json.dumps({"ops": [op], "rationale": "fake"})
        elif "choose which research persona module" in prompt:
            text = json.dumps({"route": self.rng.choice(list(PERSONAS) + ["none"])})
        elif "Predict your own behavior" in prompt:
            text = json.dumps({d: {"success": 0.4, "failure": 0.3, "abstain": 0.2, "compute": 1.8}
                               for d in ("lean_proof", "python_repair", "design_sim")})
        else:
            ids = re.findall(r"^- (a\d+|abstain) \[cost", prompt, flags=re.M)
            choice = self.rng.choice(ids)
            raw = {a: self.rng.random() for a in ids}
            raw[choice] += 1.0
            tot = sum(raw.values())
            text = json.dumps({"choice": choice, "confidence": round(self.rng.uniform(0.3, 0.9), 2),
                               "probabilities": {a: round(v / tot, 3) for a, v in raw.items()}})
        return {"id": f"fake-{self.calls}", "model": payload["model"],
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4}}
