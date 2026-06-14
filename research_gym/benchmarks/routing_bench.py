from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from research_gym.envs.routing import RoutingExample, RoutingRunResult, split_examples, tokenize


class LexicalTRMRouter:
    """Dependency-light stand-in for Tesseract's TF-IDF MLP router."""

    def __init__(self) -> None:
        self.envs: list[str] = []
        self.priors: dict[str, float] = {}
        self.token_counts: dict[str, Counter[str]] = {}
        self.totals: dict[str, int] = {}
        self.vocab: set[str] = set()

    def fit(self, examples: list[RoutingExample]) -> "LexicalTRMRouter":
        env_counts = Counter(example.env_id for example in examples)
        self.envs = sorted(env_counts)
        self.priors = {env: math.log(env_counts[env] / len(examples)) for env in self.envs}
        self.token_counts = {env: Counter() for env in self.envs}
        for example in examples:
            tokens = tokenize(example.prompt)
            self.vocab.update(tokens)
            self.token_counts[example.env_id].update(tokens)
        self.totals = {env: sum(counts.values()) for env, counts in self.token_counts.items()}
        return self

    def scores(self, prompt: str, candidates: set[str] | None = None) -> dict[str, float]:
        tokens = tokenize(prompt)
        envs = sorted(candidates) if candidates else self.envs
        denom_extra = max(1, len(self.vocab))
        scores: dict[str, float] = {}
        for env in envs:
            score = self.priors.get(env, -100.0)
            denom = self.totals.get(env, 0) + denom_extra
            counts = self.token_counts.get(env, Counter())
            for token in tokens:
                score += math.log((counts.get(token, 0) + 1) / denom)
            scores[env] = score
        return scores

    def predict(self, prompt: str, candidates: set[str] | None = None) -> str:
        scores = self.scores(prompt, candidates)
        return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


class TokenLatticeRouter:
    """LDT router: prompt tokens monotonically refine candidate env IDs."""

    def __init__(self, *, min_token_count: int = 2, dominance: float = 0.65) -> None:
        self.min_token_count = min_token_count
        self.dominance = dominance
        self.envs: set[str] = set()
        self.token_to_envs: dict[str, set[str]] = {}
        self.env_token_counts: dict[str, Counter[str]] = {}

    def fit(self, examples: list[RoutingExample]) -> "TokenLatticeRouter":
        self.envs = {example.env_id for example in examples}
        self.env_token_counts = {env: Counter() for env in self.envs}
        token_totals: dict[str, Counter[str]] = defaultdict(Counter)
        for example in examples:
            unique_tokens = set(tokenize(example.prompt))
            self.env_token_counts[example.env_id].update(unique_tokens)
            for token in unique_tokens:
                token_totals[token][example.env_id] += 1

        for token, env_counts in token_totals.items():
            total = sum(env_counts.values())
            if total < self.min_token_count:
                continue
            supported = {
                env
                for env, count in env_counts.items()
                if count >= self.min_token_count and count / total >= self.dominance
            }
            if supported:
                self.token_to_envs[token] = supported
        return self

    def candidates(self, prompt: str) -> set[str]:
        candidates = set(self.envs)
        narrowed = False
        for token in set(tokenize(prompt)):
            supported = self.token_to_envs.get(token)
            if not supported:
                continue
            refined = candidates & supported
            if refined:
                candidates = refined
                narrowed = True
        return candidates if narrowed else set(self.envs)

    def predict(self, prompt: str) -> str | None:
        candidates = self.candidates(prompt)
        if len(candidates) == 1:
            return next(iter(candidates))
        tokens = set(tokenize(prompt))
        evidence = {
            env: sum(self.env_token_counts[env].get(token, 0) for token in tokens)
            for env in candidates
        }
        if not evidence:
            return None
        best_score = max(evidence.values())
        best = [env for env, score in evidence.items() if score == best_score]
        return sorted(best)[0] if best_score > 0 else None


class HybridRoutingModel:
    def __init__(self, ldt: TokenLatticeRouter, trm: LexicalTRMRouter) -> None:
        self.ldt = ldt
        self.trm = trm

    def predict(self, prompt: str) -> tuple[str, int]:
        candidates = self.ldt.candidates(prompt)
        trm_prediction = self.trm.predict(prompt)
        if trm_prediction in candidates:
            return self.trm.predict(prompt, candidates), len(candidates)
        # Token-derived routing is model/experience evidence, not hard soundness.
        # If the lattice proposal would eliminate the TRM top route, keep it soft.
        return trm_prediction, len(candidates)


def evaluate_router(name: str, examples: list[RoutingExample], predict_fn) -> RoutingRunResult:
    correct = 0
    abstained = 0
    candidate_counts: list[int] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for example in examples:
        prediction_result = predict_fn(example.prompt)
        if isinstance(prediction_result, tuple):
            predicted, candidate_count = prediction_result
            candidate_counts.append(candidate_count)
        else:
            predicted = prediction_result
        if predicted is None:
            abstained += 1
            predicted = "<abstain>"
        correct += int(predicted == example.env_id)
        confusion[example.env_id][predicted] += 1
    total = len(examples)
    return RoutingRunResult(
        router=name,
        accuracy=correct / total if total else 0.0,
        correct=correct,
        total=total,
        abstained=abstained,
        avg_candidates=sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        confusion={target: dict(preds) for target, preds in sorted(confusion.items())},
    )


def run_routing_benchmark(examples: list[RoutingExample], *, train_ratio: float = 0.7, seed: int = 7) -> dict[str, object]:
    train, test = split_examples(examples, train_ratio=train_ratio, seed=seed)
    trm = LexicalTRMRouter().fit(train)
    ldt = TokenLatticeRouter().fit(train)
    hybrid = HybridRoutingModel(ldt, trm)
    results = [
        evaluate_router("ldt", test, ldt.predict),
        evaluate_router("trm", test, trm.predict),
        evaluate_router("hybrid", test, hybrid.predict),
    ]
    return {
        "train_size": len(train),
        "test_size": len(test),
        "envs": sorted({example.env_id for example in examples}),
        "results": [result.to_jsonable() for result in results],
    }


def summary_markdown(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, list)
    lines = [
        "# Env Pointer Routing Benchmark",
        "",
        f"Train examples: {payload['train_size']}",
        f"Test examples: {payload['test_size']}",
        "",
        "| Router | Accuracy | Correct | Abstained | Avg Candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        assert isinstance(result, dict)
        lines.append(
            f"| `{result['router']}` | {float(result['accuracy']):.3f} | "
            f"{int(result['correct'])}/{int(result['total'])} | {int(result['abstained'])} | "
            f"{float(result['avg_candidates']):.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_experiment_bundle(payload: dict[str, object], out_dir: Path, *, notes: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "training_notes.md").write_text(notes, encoding="utf-8")
