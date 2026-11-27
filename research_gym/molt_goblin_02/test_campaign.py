import json

from . import campaign as C
from . import inky as I
from .gym import TaskSpec


def _tiny(monkeypatch):
    spec = TaskSpec("cumsum", 5)
    monkeypatch.setattr(C, "PANEL", {"cumsum": {"train": spec, "eval": {"dev": spec, "ood_long": TaskSpec("cumsum", 7)}}})
    monkeypatch.setattr(C, "RUNGS", ({"steps": 16, "seeds": (1,)}, {"steps": 16, "seeds": (1, 2)}, {"steps": 16, "seeds": (1, 2)}))
    base = C.BASE.replace(width=16, loops=2, steps=16)
    monkeypatch.setattr(C, "BASE", base)
    monkeypatch.setattr(C, "BASELINES", {"prefix": base.replace(memory="prefix"), "tied": base.replace(memory="tied"),
                                         "prev": base.replace(memory="prev")})


class FakeInky:
    usage = {"requests": 0}

    def __init__(self):
        self.n = 0

    def ask(self, system, user, purpose):
        assert "label" not in user.lower() or "labels" in user.lower()
        self.n += 1
        if purpose == "adversarial_task":
            body = {"explanations": ["a", "b"], "spec": {"family": "cumsum", "length": 6, "depth": 0, "distract": 1,
                                                         "variant": "clean"}, "prediction": "x"}
        elif self.n == 2:
            return {"ok": True, "text": "no json here"}
        else:
            body = {"diagnosis": "d", "revision": "r", "hypothesis": "h", "mechanism": "1", "base": "prefix",
                    "candidate": {"memory": "full", "loss": "all" if self.n % 2 else "ramp"},
                    "control": {"memory": "prev"}, "predicted": {"cumsum": "up"}, "rationale": "z"}
        return {"ok": True, "text": "thinking\n```json\n" + json.dumps(body) + "\n```", "seconds": 0.1}


def test_campaign_loop_resumes(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "ADVERSARIAL_EVERY", 3)
    camp = C.Campaign(tmp_path, hours=1, threads=1, use_inky=False, max_iterations=4)
    camp.inky = FakeInky()
    camp.state["proposer"] = "inky"
    camp.run()
    st = json.loads((tmp_path / "state.json").read_text())
    kinds = [h["kind"] for h in st["history"]]
    assert kinds.count("proposal") == 4 and "adversarial_task" in kinds
    assert any("no JSON" in (h.get("error") or "") for h in st["history"])
    assert st["adversarial"] and st["adversarial"][0]["family"] == "cumsum"
    assert any(h.get("verdict") for h in st["history"])
    assert any(p["to"] >= 1 for p in st["promotions"])
    n_cache = len(st["cache"])
    again = C.Campaign(tmp_path, hours=1, threads=1, use_inky=False, max_iterations=4)
    assert again.state["resumed"] == 1 and len(again.state["cache"]) == n_cache


def test_fallback_proposer(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    camp = C.Campaign(tmp_path, hours=1, threads=1, use_inky=False, max_iterations=2)
    camp.run()
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["proposer"] == "deterministic_fallback"
    assert all("candidate_id" in h or h.get("error") for h in st["history"])


def test_extract_json_and_paid_guard():
    assert I.extract_json('a {"x": 1} b {"y": {"z": 2}}') == {"y": {"z": 2}}
    assert I.extract_json("```json\n{\"k\": 3}\n```") == {"k": 3}
    inky = I.Inky(transport=lambda p: {"choices": [{"message": {"content": "hi"}}], "model": "other/paid-model"})
    try:
        inky.ask("s", "u", "t")
        raise AssertionError("expected block")
    except I.PaidModelBlocked:
        pass


def test_retry_bounded(monkeypatch):
    monkeypatch.setattr(I, "RETRY_DELAYS", (0, 0))
    calls = []

    def boom(p):
        calls.append(1)
        raise TimeoutError()
    r = I.Inky(transport=boom).ask("s", "u", "t")
    assert not r["ok"] and len(calls) == 3
