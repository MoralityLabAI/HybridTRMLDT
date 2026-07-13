from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from research_gym.core.typed_soundness import SoundnessType


@dataclass(frozen=True)
class ControlTask:
    task_id: str
    split: str
    application: str
    source_repo: str
    skill: str
    actions: tuple[str, ...]
    utilities: dict[str, float]
    trm_scores: dict[str, float]
    ldt_scores: dict[str, float]
    environment_allowed: tuple[str, ...]
    soft_candidates: tuple[str, ...]
    soft_soundness: SoundnessType
    context: str

    @property
    def optimal_action(self) -> str:
        return max(self.environment_allowed, key=lambda action: (self.utilities[action], action))

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["actions"] = list(self.actions)
        payload["environment_allowed"] = list(self.environment_allowed)
        payload["soft_candidates"] = list(self.soft_candidates)
        payload["soft_soundness"] = self.soft_soundness.value
        payload["optimal_action"] = self.optimal_action
        return payload


APPLICATIONS = (
    "secret_route",
    "moral_story",
    "diplomacy_orders",
    "oracle_control",
    "bitvm_profile",
    "prisoner_dilemma",
    "memetic_treaty",
    "strategic_tick",
    "studio_skill_route",
)


def _base_utilities(actions: tuple[str, ...], optimal: str, runner_up: str) -> dict[str, float]:
    utilities = {action: 0.15 for action in actions}
    utilities[runner_up] = 0.62
    utilities[optimal] = 1.0
    return utilities


def _noisy_scores(rng: random.Random, utilities: dict[str, float], sigma: float) -> dict[str, float]:
    return {action: round(value + rng.gauss(0.0, sigma), 4) for action, value in utilities.items()}


def _task_for(application: str, index: int, split: str, rng: random.Random) -> ControlTask:
    suffix = f"{split}-{index:03d}"
    offset = index + (1000 if split == "eval" else 0)

    if application == "secret_route":
        actions = ("trace_clue", "stabilize", "rush", "wait")
        optimal = ("trace_clue", "stabilize")[offset % 2]
        runner_up = "stabilize" if optimal == "trace_clue" else "trace_clue"
        utilities = _base_utilities(actions, optimal, runner_up)
        utilities["rush"] = 0.72
        allowed = tuple(action for action in actions if action != "rush" or offset % 3 == 0)
        trm = _noisy_scores(rng, utilities, 0.18)
        trm["rush"] += 0.38
        if offset % 3 == 0:
            # Reachable shortcuts create a safe but low-margin ranking error.
            # A second certified branch can recover the exact route.
            trm[runner_up] = trm[optimal] + 0.04
            trm["rush"] = min(trm["rush"], trm[optimal] - 0.08)
        ldt = _noisy_scores(rng, utilities, 0.07)
        soft = ("rush", runner_up) if offset % 4 else (optimal, runner_up)
        return ControlTask(
            f"secret-route-{suffix}", split, application, "GPTStoryworld", "reachability",
            actions, utilities, trm, ldt, allowed, soft, SoundnessType.MODEL_SOUND_DEAD,
            "Secret-ending route with exact transition reachability and a tempting local shortcut.",
        )

    if application == "moral_story":
        actions = ("reconcile", "expose", "contain", "defer")
        optimal = actions[offset % 3]
        runner_up = actions[(offset + 1) % 3]
        utilities = _base_utilities(actions, optimal, runner_up)
        utilities["defer"] = 0.35
        trm = _noisy_scores(rng, utilities, 0.11)
        ldt = _noisy_scores(rng, {"reconcile": 0.45, "expose": 0.35, "contain": 0.9, "defer": 0.7}, 0.08)
        soft = ("contain", "defer") if offset % 4 else (optimal, "contain")
        return ControlTask(
            f"moral-story-{suffix}", split, application, "GPTStoryworld", "moral_optimization",
            actions, utilities, trm, ldt, actions, soft, SoundnessType.EXPERIENCE_SOUND_DEAD,
            "All choices are legal; morality is a preference surface rather than a hard ending gate.",
        )

    if application == "diplomacy_orders":
        actions = ("support_ally", "hold", "attack", "invalid_convoy")
        legal_strategic = ("support_ally", "hold", "attack")
        allowed = legal_strategic if offset % 5 else ("support_ally", "hold")
        optimal = allowed[offset % len(allowed)]
        runner_up = allowed[(offset + 1) % len(allowed)]
        utilities = _base_utilities(actions, optimal, runner_up)
        utilities["invalid_convoy"] = 0.0
        trm = _noisy_scores(rng, utilities, 0.17)
        if offset % 4 == 0:
            trm["invalid_convoy"] += 0.85
        ldt = _noisy_scores(rng, {"support_ally": 0.8, "hold": 0.75, "attack": 0.45, "invalid_convoy": 0.0}, 0.1)
        soft = ("support_ally", "hold") if offset % 3 else ("hold", "attack")
        return ControlTask(
            f"diplomacy-{suffix}", split, application, "AI_Diplomacy", "coalition_planning",
            actions, utilities, trm, ldt, allowed, soft, SoundnessType.MODEL_SOUND_DEAD,
            "Order legality is exact; alliance and betrayal forecasts remain opponent-model conditional.",
        )

    if application == "oracle_control":
        actions = ("remove", "retain", "escalate", "accept_raw")
        expected = ("remove", "retain", "escalate")[offset % 3]
        allowed = ("escalate",) if expected == "escalate" else (expected, "escalate")
        runner_up = "escalate" if expected != "escalate" else "remove"
        utilities = _base_utilities(actions, expected, runner_up)
        if expected != "escalate":
            utilities["escalate"] = 0.48
        utilities["accept_raw"] = 0.05
        trm = _noisy_scores(rng, utilities, 0.16)
        if offset % 3 != 1:
            trm["accept_raw"] += 1.1
        elif expected != "escalate":
            # The proposal is safe but overly cautious; provenance-aware
            # confidence arbitration should defer this narrow margin to LDT.
            trm["escalate"] = trm[expected] + 0.04
        ldt = _noisy_scores(rng, utilities, 0.04)
        soft = ("accept_raw", "retain") if offset % 3 != 1 else (expected, "escalate")
        return ControlTask(
            f"oracle-control-{suffix}", split, application, "SmallControlHarness", "provenance_control",
            actions, utilities, trm, ldt, allowed, soft, SoundnessType.UNKNOWN,
            "Signed evidence is untrusted until its current semantic anchor certifies the requested use.",
        )

    if application == "bitvm_profile":
        actions = ("minimal_patch", "attested_bundle", "watchtower", "defer_funding")
        optimal = actions[offset % len(actions)]
        allowed = actions if offset % 2 else tuple(action for action in actions if action != "minimal_patch")
        if optimal not in allowed:
            optimal = "attested_bundle"
        runner_up = "watchtower" if optimal != "watchtower" else "attested_bundle"
        utilities = _base_utilities(actions, optimal, runner_up)
        trm = _noisy_scores(rng, utilities, 0.18)
        trm["minimal_patch"] += 0.32
        ldt = _noisy_scores(
            rng,
            {"minimal_patch": 0.2, "attested_bundle": 0.95, "watchtower": 0.75, "defer_funding": 0.82},
            0.08,
        )
        soft = ("attested_bundle", "watchtower") if offset % 3 else ("watchtower", "defer_funding")
        return ControlTask(
            f"bitvm-profile-{suffix}", split, application, "BitVMArena", "control_profile_selection",
            actions, utilities, trm, ldt, allowed, soft, SoundnessType.EXPERIENCE_SOUND_DEAD,
            "Protocol compatibility is exact; attack frequency and control-profile ranking are empirical.",
        )

    if application == "prisoner_dilemma":
        actions = ("cooperate", "cautious", "defect", "wait")
        optimal = ("cooperate", "cautious", "defect")[offset % 3]
        runner_up = "cautious" if optimal != "cautious" else "cooperate"
        utilities = _base_utilities(actions, optimal, runner_up)
        trm = _noisy_scores(rng, utilities, 0.13)
        trm["defect"] += 0.12
        ldt = _noisy_scores(rng, {"cooperate": 0.8, "cautious": 0.85, "defect": 0.35, "wait": 0.25}, 0.09)
        soft = (optimal, runner_up) if offset % 4 else (runner_up, "wait")
        return ControlTask(
            f"prisoner-dilemma-{suffix}", split, application, "StoryForge", "opponent_modeling",
            actions, utilities, trm, ldt, actions, soft, SoundnessType.MODEL_SOUND_DEAD,
            "Choice gates are exact, while the best response depends on a model of the other player.",
        )

    if application == "memetic_treaty":
        actions = ("bounded_channel", "public_signal", "coalition_vote", "isolate_actor")
        allowed = actions if offset % 3 == 0 else tuple(action for action in actions if action != "public_signal")
        optimal = allowed[offset % len(allowed)]
        runner_up = "bounded_channel" if optimal != "bounded_channel" else "coalition_vote"
        utilities = _base_utilities(actions, optimal, runner_up)
        trm = _noisy_scores(rng, utilities, 0.17)
        if "public_signal" not in allowed and offset % 2 == 0:
            trm["public_signal"] += 0.85
        ldt = _noisy_scores(
            rng,
            {"bounded_channel": 0.9, "public_signal": 0.25, "coalition_vote": 0.72, "isolate_actor": 0.68},
            0.09,
        )
        soft = ("bounded_channel", "coalition_vote") if offset % 4 else ("public_signal", "isolate_actor")
        return ControlTask(
            f"memetic-treaty-{suffix}", split, application, "TheySing", "treaty_channel_control",
            actions, utilities, trm, ldt, allowed, soft, SoundnessType.MODEL_SOUND_DEAD,
            "Campaign-clock and channel permissions are exact; persuasion and counterparty response remain modeled.",
        )

    if application == "strategic_tick":
        actions = ("expand", "fortify", "raid", "consolidate")
        disallowed = "raid" if offset % 2 else "expand"
        allowed = tuple(action for action in actions if action != disallowed)
        optimal = allowed[offset % len(allowed)]
        runner_up = "fortify" if optimal != "fortify" else "consolidate"
        utilities = _base_utilities(actions, optimal, runner_up)
        trm = _noisy_scores(rng, utilities, 0.18)
        if offset % 4 == 0:
            trm[disallowed] += 0.82
        ldt = _noisy_scores(
            rng,
            {"expand": 0.58, "fortify": 0.86, "raid": 0.52, "consolidate": 0.78},
            0.1,
        )
        soft = ("fortify", "consolidate") if offset % 3 else (optimal, runner_up)
        return ControlTask(
            f"strategic-tick-{suffix}", split, application, "Blighted Galaxy", "strategic_tick_control",
            actions, utilities, trm, ldt, allowed, soft, SoundnessType.EXPERIENCE_SOUND_DEAD,
            "Resource and campaign-clock legality are exact; strategic value is replay- and opponent-dependent.",
        )

    if application == "studio_skill_route":
        actions = ("game_designer", "ai_programmer", "qa_tester", "generalist")
        specialist = actions[offset % 3]
        allowed = (specialist, "generalist")
        utilities = _base_utilities(actions, specialist, "generalist")
        utilities["generalist"] = 0.5
        trm = _noisy_scores(rng, utilities, 0.14)
        if offset % 4 == 0:
            trm["generalist"] += 0.42
        ldt = _noisy_scores(rng, utilities, 0.05)
        soft = (specialist, "generalist") if offset % 5 else ("generalist", actions[(offset + 1) % 3])
        return ControlTask(
            f"studio-skill-{suffix}", split, application, "CodexGameStudio", "skill_orchestration",
            actions, utilities, trm, ldt, allowed, soft, SoundnessType.EXPERIENCE_SOUND_DEAD,
            "Declared agent scope is exact; semantic task-to-specialist ranking is a learned routing judgment.",
        )

    raise ValueError(f"Unknown control-task application: {application}")


def generate_control_tasks(*, n_train: int = 48, n_eval: int = 48, seed: int = 23) -> list[ControlTask]:
    tasks: list[ControlTask] = []
    for application_index, application in enumerate(APPLICATIONS):
        train_rng = random.Random(seed + 101 * application_index)
        eval_rng = random.Random(seed + 10_000 + 101 * application_index)
        tasks.extend(_task_for(application, index, "train", train_rng) for index in range(n_train))
        tasks.extend(_task_for(application, index, "eval", eval_rng) for index in range(n_eval))
    return tasks
