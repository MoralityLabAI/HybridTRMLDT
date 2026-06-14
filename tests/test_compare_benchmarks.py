import json

from research_gym.scripts.compare_benchmarks import inferior_hybrid_cases, load_all, markdown_report


def test_load_all_saved_benchmarks():
    results = load_all(__import__("pathlib").Path("data/benchmarks"))

    assert set(results) == {"sudoku", "arc1", "arc2", "routing", "storyworld"}
    assert results["sudoku"]["hybrid"]["score"] == 1.0
    assert results["routing"]["trm"]["score"] == results["routing"]["hybrid"]["score"]


def test_inferior_hybrid_cases_detects_score_regression():
    results = {
        "toy": {
            "ldt": {"score": 1.0},
            "trm": {"score": 0.5},
            "hybrid": {"score": 0.75},
        }
    }

    assert inferior_hybrid_cases(results) == ["toy: hybrid score 0.750 below best 1.000"]


def test_markdown_report_mentions_core_findings():
    results = load_all(__import__("pathlib").Path("data/benchmarks"))
    report = markdown_report(results)

    assert "Where TRM Is Effective" in report
    assert "No aggregate benchmark has hybrid below the best score" in report
    assert "routing" in report
