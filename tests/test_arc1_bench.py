from research_gym.benchmarks.arc_bench import (
    run_arc1_benchmark,
    solve_arc1_hybrid,
    solve_arc1_ldt,
    solve_arc1_trm,
)
from research_gym.envs.arc_tasks import (
    default_arc1_tasks,
    fill_zero_with,
    flip_h,
    grid,
    map_color,
    primitive_rules,
    rule_fits,
)


def test_arc_primitive_rules():
    g = grid([[1, 0], [2, 1]])

    assert map_color(g, 1, 3) == grid([[3, 0], [2, 3]])
    assert fill_zero_with(g, 9) == grid([[1, 9], [2, 1]])
    assert flip_h(grid([[1, 2, 3]])) == grid([[3, 2, 1]])


def test_default_tasks_have_matching_named_rules():
    rules = {rule.name: rule for rule in primitive_rules()}

    for task in default_arc1_tasks():
        assert rule_fits(rules[task.rule_name], task)
        assert rules[task.rule_name].apply(task.test_input) == task.test_output


def test_arc1_solvers_solve_default_tasks():
    for task in default_arc1_tasks():
        ldt = solve_arc1_ldt(task)
        trm = solve_arc1_trm(task)
        hybrid = solve_arc1_hybrid(task)

        assert ldt.solved
        assert trm.solved
        assert hybrid.solved
        assert ldt.predicted == task.test_output
        assert trm.predicted == task.test_output
        assert hybrid.predicted == task.test_output


def test_hybrid_rejects_non_fitting_proposals():
    task = default_arc1_tasks()[2]
    result = solve_arc1_hybrid(task)

    assert result.solved
    assert result.rejected >= 1
    assert any(item.startswith("reject:") for item in result.trace)


def test_arc1_benchmark_shape():
    results = run_arc1_benchmark()

    assert len(results) == 9
    assert {result.solver for result in results} == {"ldt", "trm", "hybrid"}
    assert {result.task_id for result in results} == {task.task_id for task in default_arc1_tasks()}
