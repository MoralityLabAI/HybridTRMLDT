from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Mapping

import verifiers.v1 as vf


SequencerName = Literal[
    "global_signed",
    "lineage_only",
    "fixed_typed",
    "control_math",
    "local_calibrated",
    "airis_das",
]


class HybridSequencerTasksetConfig(vf.TasksetConfig):
    data_path: str = "data/replay_tasks.jsonl"
    family: str | None = None
    limit: int | None = None


class HybridSequencerHarnessConfig(vf.HarnessConfig):
    sequencer: SequencerName = "control_math"
    max_turns: int = 1


class HybridSequencerTaskset(vf.Taskset):
    config_type = HybridSequencerTasksetConfig


class HybridSequencerHarness(vf.Harness):
    config_type = HybridSequencerHarnessConfig


def _data_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).parent / path


def _source(config: HybridSequencerTasksetConfig):
    path = _data_path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"sequencer replay taskset not found: {path}")
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if config.family is not None and row["family"] != config.family:
                continue
            yield row
            emitted += 1
            if config.limit is not None and emitted >= config.limit:
                break


@vf.reward(weight=1.0)
async def sequencer_utility(task, state) -> float:
    return float(state["selected_outcome"]["utility"])


@vf.metric
async def sequencer_correct(task, state) -> float:
    return float(state["selected_outcome"]["correct"])


@vf.metric
async def sequencer_cost(task, state) -> float:
    return float(state["selected_outcome"]["cost"])


@vf.metric
async def sequencer_constraint_violation(task, state) -> float:
    return float(state["selected_outcome"]["constraint_violation"])


@vf.metric
async def signed_control_bound(task, state) -> float:
    return float(task["topology"]["signed_control_loss_upper_bound"])


def _program(sequencer: SequencerName):
    async def replay(task, state):
        sequence = str(task["selections"][sequencer])
        candidate_outcomes = task["candidate_outcomes"]
        if not isinstance(candidate_outcomes, Mapping) or sequence not in candidate_outcomes:
            raise ValueError(f"missing selected sequence outcome: {sequence}")
        outcome = dict(candidate_outcomes[sequence])
        state["sequencer"] = sequencer
        state["selected_sequence"] = sequence
        state["selected_outcome"] = outcome
        state["answer"] = outcome["answer"]
        state["completion"] = [{"role": "assistant", "content": str(outcome["answer"])}]
        state["control_receipt"] = {
            "context": task["context"],
            "topology": task["topology"],
            "selected_sequence": sequence,
        }
        if sequencer == "airis_das":
            receipt = task.get("airis_das")
            if not isinstance(receipt, Mapping):
                raise ValueError("AIRIS/DAS replay requires a sealed bridge receipt")
            state["control_receipt"]["airis_das"] = receipt
        state.stop("sequencer_replayed")
        return state

    return replay


def load_taskset(config: HybridSequencerTasksetConfig) -> HybridSequencerTaskset:
    resolved = HybridSequencerTasksetConfig(config)
    return HybridSequencerTaskset(
        source=lambda: _source(resolved),
        rewards=[sequencer_utility],
        metrics=[
            sequencer_correct,
            sequencer_cost,
            sequencer_constraint_violation,
            signed_control_bound,
        ],
        config=resolved,
    )


def load_harness(config: HybridSequencerHarnessConfig) -> HybridSequencerHarness:
    resolved = HybridSequencerHarnessConfig(config)
    return HybridSequencerHarness(program=_program(resolved.sequencer), config=resolved)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(HybridSequencerTasksetConfig(config.taskset)),
        harness=load_harness(HybridSequencerHarnessConfig(config.harness)),
    )
