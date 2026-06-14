from __future__ import annotations

from collections import defaultdict

from research_gym.envs.arc_tasks import (
    ArcRule,
    ArcRunResult,
    ArcTask,
    default_arc1_tasks,
    default_arc2_tasks,
    primitive_rule_pairs,
    primitive_rules,
    rule_fits,
)


def _ldt_candidate_families(task: ArcTask, rules: list[ArcRule]) -> set[str]:
    input_colors = {cell for example in task.train for row in example.input_grid for cell in row}
    output_colors = {cell for example in task.train for row in example.output_grid for cell in row}
    if input_colors != output_colors:
        return {"color_map", "fill"}
    return {"spatial"}


def solve_arc1_ldt(task: ArcTask, rules: list[ArcRule] | None = None) -> ArcRunResult:
    rules = rules or primitive_rules()
    families = _ldt_candidate_families(task, rules)
    candidates = [rule for rule in rules if rule.family in families]
    trace = [f"family_candidates:{','.join(sorted(families))}"]
    fitting = [rule for rule in candidates if rule_fits(rule, task)]
    trace.extend(f"fit:{rule.name}" for rule in fitting)
    if len(fitting) != 1:
        return ArcRunResult(
            solver="ldt",
            task_id=task.task_id,
            solved=False,
            predicted=task.test_input,
            expected=task.test_output,
            steps=len(candidates),
            proposals=0,
            rejected=0,
            trace=trace + ["abstain:ambiguous_or_no_rule"],
        )
    rule = fitting[0]
    predicted = rule.apply(task.test_input)
    return ArcRunResult(
        solver="ldt",
        task_id=task.task_id,
        solved=predicted == task.test_output,
        predicted=predicted,
        expected=task.test_output,
        steps=len(candidates),
        proposals=0,
        rejected=0,
        trace=trace + [f"apply:{rule.name}"],
    )


def solve_arc1_trm(task: ArcTask, rules: list[ArcRule] | None = None) -> ArcRunResult:
    rules = rules or primitive_rules()
    trace: list[str] = []
    rejected = 0
    for step, rule in enumerate(rules, start=1):
        trace.append(f"try:{rule.name}")
        if rule_fits(rule, task):
            predicted = rule.apply(task.test_input)
            return ArcRunResult(
                solver="trm",
                task_id=task.task_id,
                solved=predicted == task.test_output,
                predicted=predicted,
                expected=task.test_output,
                steps=step,
                proposals=step,
                rejected=rejected,
                trace=trace + [f"apply:{rule.name}"],
            )
        rejected += 1
    return ArcRunResult(
        solver="trm",
        task_id=task.task_id,
        solved=False,
        predicted=task.test_input,
        expected=task.test_output,
        steps=len(rules),
        proposals=len(rules),
        rejected=rejected,
        trace=trace + ["fail:no_rule"],
    )


def solve_arc1_hybrid(task: ArcTask, rules: list[ArcRule] | None = None) -> ArcRunResult:
    rules = rules or primitive_rules()
    families = _ldt_candidate_families(task, rules)
    proposals = [rule for rule in rules if rule.family in families]
    trace = [f"certified_families:{','.join(sorted(families))}"]
    rejected = 0
    for step, rule in enumerate(proposals, start=1):
        trace.append(f"propose:{rule.name}")
        if not rule_fits(rule, task):
            rejected += 1
            trace.append(f"reject:{rule.name}")
            continue
        predicted = rule.apply(task.test_input)
        return ArcRunResult(
            solver="hybrid",
            task_id=task.task_id,
            solved=predicted == task.test_output,
            predicted=predicted,
            expected=task.test_output,
            steps=step,
            proposals=step,
            rejected=rejected,
            trace=trace + [f"apply:{rule.name}"],
        )
    return ArcRunResult(
        solver="hybrid",
        task_id=task.task_id,
        solved=False,
        predicted=task.test_input,
        expected=task.test_output,
        steps=len(proposals),
        proposals=len(proposals),
        rejected=rejected,
        trace=trace + ["fail:no_certified_rule"],
    )


def run_arc1_benchmark(tasks: list[ArcTask] | None = None) -> list[ArcRunResult]:
    tasks = tasks or default_arc1_tasks()
    results: list[ArcRunResult] = []
    for task in tasks:
        results.extend([solve_arc1_ldt(task), solve_arc1_trm(task), solve_arc1_hybrid(task)])
    return results


def _pair_family_allowed(task: ArcTask, rule: ArcRule) -> bool:
    families = _ldt_candidate_families(task, primitive_rules())
    parts = set(rule.family.split("+"))
    if "spatial" in families and "spatial" not in parts:
        return False
    if {"color_map", "fill"} & families and ({"color_map", "fill"} & parts):
        return True
    return "spatial" in families and "spatial" in parts


def _unique_prediction(fitting: list[ArcRule], task: ArcTask):
    predictions = {rule.apply(task.test_input) for rule in fitting}
    if len(predictions) == 1:
        return next(iter(predictions))
    return None


def solve_arc2_ldt(task: ArcTask, rules: list[ArcRule] | None = None) -> ArcRunResult:
    rules = rules or primitive_rule_pairs()
    candidates = [rule for rule in rules if _pair_family_allowed(task, rule)]
    trace = [f"pair_candidates:{len(candidates)}"]
    fitting = [rule for rule in candidates if rule_fits(rule, task)]
    trace.extend(f"fit:{rule.name}" for rule in fitting[:4])
    predicted = _unique_prediction(fitting, task)
    if predicted is None:
        return ArcRunResult(
            solver="ldt",
            task_id=task.task_id,
            solved=False,
            predicted=task.test_input,
            expected=task.test_output,
            steps=len(candidates),
            proposals=0,
            rejected=0,
            trace=trace + ["abstain:ambiguous_or_no_pair"],
        )
    return ArcRunResult(
        solver="ldt",
        task_id=task.task_id,
        solved=predicted == task.test_output,
        predicted=predicted,
        expected=task.test_output,
        steps=len(candidates),
        proposals=0,
        rejected=0,
        trace=trace + [f"apply:certified_equivalence:{len(fitting)}"],
    )


def solve_arc2_trm(task: ArcTask, rules: list[ArcRule] | None = None) -> ArcRunResult:
    rules = rules or primitive_rule_pairs()
    rejected = 0
    trace: list[str] = []
    for step, rule in enumerate(rules, start=1):
        trace.append(f"try:{rule.name}")
        fitting = [candidate for candidate in rules if rule_fits(candidate, task)]
        if rule_fits(rule, task):
            predicted = _unique_prediction(fitting, task) or rule.apply(task.test_input)
            return ArcRunResult(
                solver="trm",
                task_id=task.task_id,
                solved=predicted == task.test_output,
                predicted=predicted,
                expected=task.test_output,
                steps=step,
                proposals=step,
                rejected=rejected,
                trace=trace + [f"apply:{rule.name}"],
            )
        rejected += 1
    return ArcRunResult(
        solver="trm",
        task_id=task.task_id,
        solved=False,
        predicted=task.test_input,
        expected=task.test_output,
        steps=len(rules),
        proposals=len(rules),
        rejected=rejected,
        trace=trace + ["fail:no_pair"],
    )


def solve_arc2_hybrid(task: ArcTask, rules: list[ArcRule] | None = None) -> ArcRunResult:
    rules = rules or primitive_rule_pairs()
    proposals = [rule for rule in rules if _pair_family_allowed(task, rule)]
    proposals = sorted(proposals, key=lambda rule: (len(set(rule.family.split("+"))) == 1, rule.name))
    rejected = 0
    trace = [f"certified_pair_candidates:{len(proposals)}"]
    fitting: list[ArcRule] = []
    for step, rule in enumerate(proposals, start=1):
        trace.append(f"propose:{rule.name}")
        if not rule_fits(rule, task):
            rejected += 1
            continue
        fitting.append(rule)
        predicted = _unique_prediction(fitting, task)
        if predicted is None:
            rejected += 1
            trace.append(f"defer:ambiguous:{len(fitting)}")
            continue
        return ArcRunResult(
            solver="hybrid",
            task_id=task.task_id,
            solved=predicted == task.test_output,
            predicted=predicted,
            expected=task.test_output,
            steps=step,
            proposals=step,
            rejected=rejected,
            trace=trace + [f"apply:{rule.name}"],
        )
    return ArcRunResult(
        solver="hybrid",
        task_id=task.task_id,
        solved=False,
        predicted=task.test_input,
        expected=task.test_output,
        steps=len(proposals),
        proposals=len(proposals),
        rejected=rejected,
        trace=trace + ["fail:no_certified_pair"],
    )


def run_arc2_benchmark(tasks: list[ArcTask] | None = None) -> list[ArcRunResult]:
    tasks = tasks or default_arc2_tasks()
    results: list[ArcRunResult] = []
    for task in tasks:
        results.extend([solve_arc2_ldt(task), solve_arc2_trm(task), solve_arc2_hybrid(task)])
    return results


def summarize_results(results: list[ArcRunResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[ArcRunResult]] = defaultdict(list)
    for result in results:
        grouped[result.solver].append(result)
    summary: dict[str, dict[str, float]] = {}
    for solver, items in sorted(grouped.items()):
        count = len(items)
        summary[solver] = {
            "tasks": float(count),
            "solved": float(sum(item.solved for item in items)),
            "solve_rate": sum(item.solved for item in items) / count if count else 0.0,
            "steps": float(sum(item.steps for item in items)),
            "proposals": float(sum(item.proposals for item in items)),
            "rejected": float(sum(item.rejected for item in items)),
        }
    return summary


def summary_markdown(results: list[ArcRunResult], *, title: str = "ARC-1 Benchmark", description: str | None = None) -> str:
    summary = summarize_results(results)
    description = description or "Single-rule grid transformation benchmark comparing LDT, TRM, and hybrid rule selection."
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "| Solver | Solved | Solve Rate | Steps | Proposals | Rejected |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for solver, metrics in summary.items():
        lines.append(
            f"| `{solver}` | {int(metrics['solved'])}/{int(metrics['tasks'])} | "
            f"{metrics['solve_rate']:.3f} | {int(metrics['steps'])} | "
            f"{int(metrics['proposals'])} | {int(metrics['rejected'])} |"
        )
    lines.extend(["", "## Per Task", ""])
    for result in results:
        lines.append(
            f"- `{result.task_id}` `{result.solver}` solved={result.solved} "
            f"steps={result.steps} proposals={result.proposals} rejected={result.rejected}"
        )
    return "\n".join(lines) + "\n"
