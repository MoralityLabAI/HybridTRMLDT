from research_gym.benchmarks.arc_bench import (
    run_arc2_benchmark,
    solve_arc2_hybrid,
    solve_arc2_ldt,
    solve_arc2_trm,
)
from research_gym.envs.arc_tasks import default_arc2_tasks, grid, primitive_rule_pairs, rule_fits


def test_default_arc2_tasks_have_matching_compositions():
    rules = {rule.name: rule for rule in primitive_rule_pairs()}

    for task in default_arc2_tasks():
        assert task.rule_name in rules
        assert rule_fits(rules[task.rule_name], task)
        assert rules[task.rule_name].apply(task.test_input) == task.test_output


def test_arc2_composed_rule_applies_in_order():
    rules = {rule.name: rule for rule in primitive_rule_pairs()}
    rule = rules["map_2_to_3+flip_h"]

    assert rule.apply(grid([[2, 1, 0]])) == grid([[0, 1, 3]])


def test_arc2_solvers_solve_default_tasks():
    for task in default_arc2_tasks():
        ldt = solve_arc2_ldt(task)
        trm = solve_arc2_trm(task)
        hybrid = solve_arc2_hybrid(task)

        assert ldt.solved
        assert trm.solved
        assert hybrid.solved


def test_arc2_hybrid_uses_fewer_proposals_than_trm_on_suite():
    results = run_arc2_benchmark()
    trm_proposals = sum(result.proposals for result in results if result.solver == "trm")
    hybrid_proposals = sum(result.proposals for result in results if result.solver == "hybrid")

    assert hybrid_proposals < trm_proposals


def test_arc2_benchmark_shape():
    results = run_arc2_benchmark()

    assert len(results) == 9
    assert {result.solver for result in results} == {"ldt", "trm", "hybrid"}
    assert {result.task_id for result in results} == {task.task_id for task in default_arc2_tasks()}
