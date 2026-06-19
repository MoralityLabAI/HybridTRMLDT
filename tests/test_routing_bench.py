from research_gym.benchmarks.routing_bench import (
    ConfidenceArbitrationRoutingModel,
    LexicalTRMRouter,
    TokenLatticeRouter,
    run_routing_benchmark,
    tune_confidence_gamma,
)
from research_gym.envs.routing import RoutingExample, split_examples, tokenize


def synthetic_examples() -> list[RoutingExample]:
    rows = []
    specs = {
        "sudoku": ["grid row column box digit", "candidate cell row solve", "number puzzle box row"],
        "arc": ["colored grid transform pattern", "input output color map", "shape grid rotate color"],
        "logic": ["truth statement verifier answer", "deduce premise conclusion", "logic puzzle theorem"],
    }
    for env, prompts in specs.items():
        for prompt in prompts * 4:
            rows.append(RoutingExample(prompt=prompt, env_id=env))
    return rows


def test_tokenize_keeps_env_like_terms():
    assert tokenize("INTELLECT_3 logic-env route #42") == ["intellect_3", "logic", "env", "route", "42"]


def test_split_examples_preserves_envs():
    train, test = split_examples(synthetic_examples(), train_ratio=0.5, seed=1)

    assert {row.env_id for row in train} == {"arc", "logic", "sudoku"}
    assert {row.env_id for row in test} == {"arc", "logic", "sudoku"}


def test_lexical_trm_router_predicts_prompt_family():
    train, _ = split_examples(synthetic_examples(), train_ratio=0.8, seed=2)
    router = LexicalTRMRouter().fit(train)

    assert router.predict("solve row column digit") == "sudoku"
    assert router.predict("rotate colored grid") == "arc"
    assert router.margin("solve row column digit") > 0.0


def test_token_lattice_router_narrows_candidates():
    train, _ = split_examples(synthetic_examples(), train_ratio=0.8, seed=2)
    router = TokenLatticeRouter(min_token_count=1, dominance=0.6).fit(train)

    candidates = router.candidates("colored grid transform")

    assert candidates == {"arc"}
    assert router.predict("premise theorem conclusion") == "logic"


def test_run_routing_benchmark_returns_three_models():
    payload = run_routing_benchmark(synthetic_examples(), train_ratio=0.7, seed=3)
    results = {result["router"]: result for result in payload["results"]}

    assert set(results) == {"ldt", "trm", "hybrid"}
    assert {result["router"] for result in payload["ablations"]} == {
        "hybrid_confidence_arbitration",
        "hybrid_hard_filter",
    }
    assert {result["router"] for result in payload["architecture_variants"]} == {
        "confidence_arbitration",
        "hard_gate",
        "typed_membrane",
    }
    assert results["trm"]["accuracy"] >= 0.8
    assert results["hybrid"]["accuracy"] >= 0.8


def test_confidence_arbitration_uses_gamma_threshold():
    train, _ = split_examples(synthetic_examples(), train_ratio=0.8, seed=2)
    trm = LexicalTRMRouter().fit(train)
    ldt = TokenLatticeRouter(min_token_count=1, dominance=0.6).fit(train)
    model = ConfidenceArbitrationRoutingModel(ldt, trm, gamma=100.0)

    predicted, candidates = model.predict("rotate colored grid")

    assert predicted == "arc"
    assert candidates == 1


def test_confidence_gamma_tuning_is_deterministic():
    train, _ = split_examples(synthetic_examples(), train_ratio=0.8, seed=2)
    trm = LexicalTRMRouter().fit(train)
    ldt = TokenLatticeRouter(min_token_count=1, dominance=0.6).fit(train)

    assert tune_confidence_gamma(train, ldt, trm) == tune_confidence_gamma(train, ldt, trm)
