import json

from . import campaign3 as C
from . import train3
from .factory3 import Config
from .gym3 import TaskSpec


def _tiny(monkeypatch, tmp_path):
    spec = TaskSpec("proof_resolve", 6, depth=2)
    monkeypatch.setattr(C, "PANEL", {"proof_resolve": {"train": spec, "eval": {"dev": spec, "ood_deep": TaskSpec("proof_resolve", 8, depth=3)}}})
    monkeypatch.setattr(C, "CAMPAIGN_FAMILIES", ("proof_resolve",))
    monkeypatch.setattr(C, "RUNGS", ({"steps": 16, "seeds": (11,)}, {"steps": 16, "seeds": (11, 23)}, {"steps": 16, "seeds": (11, 23)}))
    base = Config(width=16, loops=2, steps=16)
    hrm = base.replace(routing="hrm", n_types=4)
    monkeypatch.setattr(C, "BASE", base)
    monkeypatch.setattr(C, "TEACHER", hrm)
    monkeypatch.setattr(C, "ARMS", {"A_single": base, "B_hrm": hrm, "K_distill": base.replace(distill="logits_all", distill_coef=1.0)})
    monkeypatch.setattr(C, "SEED_ARMS", ("A_single", "B_hrm", "K_distill"))
    monkeypatch.setattr(train3, "TEACHER_CACHE", tmp_path / "teachers")


class FakeInky:
    usage = {}

    def __init__(self):
        self.n = 0

    def ask(self, system, user, purpose, effort=None):
        self.n += 1
        if purpose.startswith("adversarial"):
            body = {"explanations": ["a", "b"], "spec": {"family": "proof_resolve", "length": 7, "depth": 2}, "prediction": "p"}
        elif self.n == 2:
            return {"ok": True, "text": "nothing", "finish_reason": "stop"}
        else:
            body = {"diagnosis": "d", "revision": "r", "hypothesis": "h", "mechanism": "sparsity", "base": "B_hrm",
                    "candidate": {"routing": "sparse_st", "load_balance": 0.05 * self.n}, "control": {"routing": "sparse_st"},
                    "predicted": {"proof_resolve": "up"}, "rationale": "z"}
        return {"ok": True, "text": "```json\n" + json.dumps(body) + "\n```", "finish_reason": "stop", "seconds": 0.1}


def test_campaign3_loop(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    monkeypatch.setattr(C, "ADVERSARIAL_EVERY", 3)
    camp = C.Campaign(tmp_path / "run", hours=1, threads=1, use_inky=False, max_iterations=4)
    camp.inky = FakeInky()
    camp.state["proposer"] = "inky"
    camp.run()
    st = json.loads((tmp_path / "run" / "state.json").read_text())
    kinds = [h["kind"] for h in st["history"]]
    assert kinds.count("proposal") == 4 and "adversarial_task" in kinds
    assert st["adversarial"], st["rejected_adversarial"]
    assert st["mech"], "mechanistic summary for typed arms"
    assert any(p["to"] >= 1 for p in st["promotions"])
    assert list((tmp_path / "teachers").glob("*.pt")), "B run cached as the distillation teacher"
    # cached cells are never retrained
    calls = []
    monkeypatch.setattr(C, "_job", lambda p: calls.append(p) or ({"status": "error"}, None))
    camp._ensure([tuple(k.split("|")[:2]) + (int(k.split("|")[2]), int(k.split("|")[3])) for k in st["cache"]])
    assert calls == []
    prompt = camp._proposal_prompt()
    assert "ALREADY TESTED" in prompt and "load_balance" in prompt
    assert "research_plan" not in prompt


def test_canonical_duplicate_rejected(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    camp = C.Campaign(tmp_path / "run", hours=1, threads=1, use_inky=False, max_iterations=0)
    camp.run()

    class Dup(FakeInky):
        def ask(self, system, user, purpose, effort=None):
            body = {"mechanism": "routing", "base": "A_single", "candidate": {"st_tau": 0.3}, "control": {}}
            return {"ok": True, "text": json.dumps(body), "finish_reason": "stop"}
    camp.inky = Dup()
    camp.step()
    assert "duplicate" in camp.state["history"][-1]["error"]
