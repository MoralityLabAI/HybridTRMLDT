"""Architecture instances and source-level extraction provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .schedule import GradientMask, Word


class ProvenanceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FILL = "FILL"


@dataclass(frozen=True)
class Provenance:
    status: ProvenanceStatus
    source: str
    locator: str = ""
    sha256: str = ""
    note: str = ""


@dataclass(frozen=True)
class Cell:
    value: Any
    provenance: Provenance


@dataclass(frozen=True)
class InstanceExtraction:
    name: str
    cells: Mapping[str, Cell]

    @property
    def complete(self) -> bool:
        return all(cell.provenance.status is ProvenanceStatus.VERIFIED for cell in self.cells.values())


@dataclass(frozen=True)
class ResidualSpec:
    alpha: float
    beta: float
    exponent: float | None = None

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta < 0:
            raise ValueError("residual alpha must be positive and beta non-negative")


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    residual_sublayers: int
    parameter_count: int | None
    flops_per_token: int | None
    residual: ResidualSpec

    def __post_init__(self) -> None:
        if self.residual_sublayers < 0:
            raise ValueError("residual sublayer count must be non-negative")


@dataclass(frozen=True)
class SupervisionPoint:
    visit_position: int
    name: str = "loss"


@dataclass(frozen=True)
class CarrySpec:
    persistent_registers: frozenset[str] = field(default_factory=frozenset)
    detached_registers: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ArchitectureInstance:
    name: str
    states: frozenset[str]
    modules: Mapping[str, ModuleSpec]
    word: Word
    gradient_mask: GradientMask
    supervision: tuple[SupervisionPoint, ...]
    carry: CarrySpec = CarrySpec()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        schedule_modules = {visit.symbol.module for visit in self.word.expand()}
        missing = schedule_modules - set(self.modules)
        if missing:
            raise ValueError(f"schedule references unknown modules: {sorted(missing)}")
        visit_count = len(self.word.expand())
        if any(point.visit_position not in range(visit_count) for point in self.supervision):
            raise ValueError("supervision point is outside the schedule")


def _verified(source: str, locator: str, sha256: str, note: str = "") -> Provenance:
    return Provenance(ProvenanceStatus.VERIFIED, source, locator, sha256, note)


def _fill(note: str) -> Provenance:
    return Provenance(ProvenanceStatus.FILL, "", note=note)


def _cells(source: str, sha256: str, values: Mapping[str, tuple[str, str]]) -> dict[str, Cell]:
    return {
        key: Cell(value, _verified(source, locator, sha256))
        for key, (value, locator) in values.items()
    }


def extraction_registry() -> tuple[InstanceExtraction, ...]:
    """Return source-backed rows; unresolved cells remain explicit ``FILL`` values."""

    deeploop = "https://arxiv.org/abs/2607.13491"
    hrm = "https://github.com/sapientinc/HRM@ac15626f8db096a63c775b84c9dc868776a6feda"
    hrm_sha = "756631b3c825acc97bae655876d4acdc8416b7bc8cc17531c44b6265a4474e92"
    hrm_text = "D:/Research_Engine/forks/HRM-Text@01ef01cab5b746c8419c602424ba2d49f477ab21"
    hrm_text_sha = "db73f328451eeed9132de8c5689f7dee53b85526faf559355dc66b1aab248a19"
    trm = "https://github.com/SamsungSAILMontreal/TinyRecursiveModels@c01103738605ba39d1430519b1ee0c62f4c707f8"
    trm_sha = "05d37524062ac6b7d958dbd7a173f3f7c99eaaa0b60cf78c57638a82ad185b46"
    gym_trm = "research_gym/neural/trm.py@89a1f69"
    gym_trm_sha = "53ecbe835b3885b62b4e1ab32c42dc690f5b6d67b1561b90ff7f7983b0c2b0cf"
    gym_lattice = "research_gym/core/lattice.py@89a1f69"
    gym_lattice_sha = "58c32190bdcd1e3b4dbdfdf72851cfdf863e0993e43c05b8479d77ff3077961e"
    gym_hybrid = "research_gym/core/hybrid.py@89a1f69"
    gym_hybrid_sha = "ea779956590ad72ba5d7943f30fbf2e8f7c8c357b357808b70a9eadd5f91363f"

    rows: list[InstanceExtraction] = []
    rows.append(
        InstanceExtraction(
            "DeepLoop-GPT",
            _cells(
                deeploop,
                "paper",
                {
                    "S": ("{stream}", "architecture definition"),
                    "Phi": ("K tied Transformer blocks, J=2 each", "architecture definition"),
                    "w": ("(B1...BK)^R", "architecture definition"),
                    "input": ("stream initialization", "architecture definition"),
                    "g": ("full", "analysis setting"),
                    "sigma": ("final stream", "architecture definition"),
                    "pi": ("not applicable", "architecture definition"),
                    "rho": ("alpha=(2N)^1/2; beta=(8N)^-1/2", "residual parameterization"),
                },
            ),
        )
    )
    rows.append(
        InstanceExtraction(
            "HRM (official implementation)",
            _cells(
                hrm,
                hrm_sha,
                {
                    "S": ("{z_H,z_L}", "models/hrm/hrm_act_v1.py:168-178"),
                    "Phi": ("distinct H_level and L_level Transformer stacks; J=2/layer", "models/hrm/hrm_act_v1.py:60-99,124-127"),
                    "w": ("(L^L_cycles.H)^H_cycles", "models/hrm/hrm_act_v1.py:188-204"),
                    "input": ("input embeddings injected into every L visit", "models/hrm/hrm_act_v1.py:185-203"),
                    "g": ("only final L and final H visits", "models/hrm/hrm_act_v1.py:188-204"),
                    "sigma": ("final z_H to LM and Q heads per ACT segment", "models/hrm/hrm_act_v1.py:206-213,248-254"),
                    "pi": ("z_H,z_L carried and detached between ACT segments", "models/hrm/hrm_act_v1.py:168-178,207"),
                    "rho": ("unit residual additions followed by RMSNorm", "models/hrm/hrm_act_v1.py:77-83"),
                },
            ),
        )
    )
    hrm_text_cells = _cells(
        hrm_text,
        hrm_text_sha,
        {
            "S": ("{z_H,z_L}", "models/baselines/hrm_nocarry_bp_warmup.py:69-76"),
            "Phi": ("distinct H_level and L_level Transformer stacks; J=2/layer", "models/baselines/hrm_nocarry_bp_warmup.py:54-57; models/transformer.py:65-126"),
            "w": ("(L^L_cycles.H)^H_cycles", "models/baselines/hrm_nocarry_bp_warmup.py:83-89"),
            "input": ("z_H initialized from x; no later raw-x injection", "models/baselines/hrm_nocarry_bp_warmup.py:75-89"),
            "g": ("H-prioritized module tails from bp_steps in [2,5]", "models/baselines/hrm_nocarry_bp_warmup.py:78-89"),
            "sigma": ("final z_H passed to LM head and token CE", "models/baselines/hrm_nocarry_bp_warmup.py:91; models/lm_head.py:34-72"),
            "pi": ("no carry", "models/baselines/hrm_nocarry_bp_warmup.py:75,91,96-97"),
            "rho": ("unit residual additions; output initialization depends on config", "models/transformer.py:49-62,90-96"),
        },
    )
    rows.append(InstanceExtraction("HRM-Text bp_warmup", hrm_text_cells))
    rows.append(
        InstanceExtraction(
            "TRM (official implementation)",
            _cells(
                trm,
                trm_sha,
                {
                    "S": ("{z_H,z_L}", "models/recursive_reasoning/trm.py:184-194"),
                    "Phi": ("one tied L_level Transformer stack used for both roles; J=2/layer", "models/recursive_reasoning/trm.py:65-115,149-154"),
                    "w": ("(L_low^L_cycles.L_high)^H_cycles", "models/recursive_reasoning/trm.py:204-216"),
                    "input": ("input embeddings injected into every low-state visit", "models/recursive_reasoning/trm.py:201-216"),
                    "g": ("only final low and final high applications", "models/recursive_reasoning/trm.py:207-216"),
                    "sigma": ("final z_H to LM and Q heads per ACT segment", "models/recursive_reasoning/trm.py:218-222"),
                    "pi": ("z_H,z_L carried and detached between ACT segments", "models/recursive_reasoning/trm.py:184-194,219"),
                    "rho": ("unit residual additions followed by RMSNorm", "models/recursive_reasoning/trm.py:90-104"),
                },
            ),
        )
    )
    rows.append(
        InstanceExtraction(
            "Gym TRMProposer",
            _cells(
                gym_trm,
                gym_trm_sha,
                {
                    "S": ("{encoded,hidden}", "lines 92-104"),
                    "Phi": ("one tied GRUCell", "lines 57-65"),
                    "w": ("GRU^recurrence_steps", "lines 92-104"),
                    "input": ("encoded input read at every recurrence", "lines 95-100"),
                    "g": ("full when training; propose() is inference-only", "lines 92-110,125-137"),
                    "sigma": ("final hidden to four proposal heads", "lines 104-110"),
                    "pi": ("none across calls", "lines 92-104"),
                    "rho": ("GRU gates; no residual alpha/beta form", "lines 61,92-100"),
                },
            ),
        )
    )
    ldt_cells = _cells(
        gym_lattice,
        gym_lattice_sha,
        {
            "S": ("named interval/product or powerset lattice", "lines 7-85; core/hybrid.py:20-70"),
            "Phi": ("parameter-free meet/join/refinement operators", "lines 26-37,58-74"),
            "w": ("caller-defined symbolic refinement sequence", "core/hybrid.py:257-330"),
            "input": ("current lattice plus proposal", "core/hybrid.py:257-264"),
            "g": ("not applicable: no autograd module", "lines 1-85"),
            "sigma": ("not applicable: no loss in lattice implementation", "lines 1-85"),
            "pi": ("caller retains returned CandidateState", "core/hybrid.py:257-330"),
            "rho": ("not applicable: symbolic, parameter-free", "lines 1-85"),
        },
    )
    rows.append(InstanceExtraction("Gym LDT", ldt_cells))
    hybrid_cells = {
        "S": Cell("TRM hidden plus CandidateState lattice", _verified(gym_hybrid, "lines 20-82,257-330", gym_hybrid_sha)),
        "Phi": Cell("TRMProposer plus parameter-free certification membrane", _verified(gym_hybrid, "lines 257-330; neural/trm.py:36-110", gym_hybrid_sha)),
        "w": Cell("TRM recurrence then certify_and_apply", _verified(gym_hybrid, "lines 257-330; neural/trm.py:92-110", gym_hybrid_sha)),
        "input": Cell("story/candidate features to TRM; proposal/current state to membrane", _verified(gym_hybrid, "lines 257-264; neural/trm.py:71-100", gym_hybrid_sha)),
        "g": Cell("TRM training mask is caller-defined; membrane has no gradient", _fill("no single hybrid training loop defines this cell")),
        "sigma": Cell("TRM head losses are benchmark-specific", _fill("no single hybrid supervision rule in core implementation")),
        "pi": Cell("CandidateState returned to caller; neural hidden is not carried", _verified(gym_hybrid, "lines 257-330; neural/trm.py:92-110", gym_hybrid_sha)),
        "rho": Cell("TRM GRU has no residual alpha/beta; membrane is symbolic", _verified(gym_hybrid, "neural/trm.py:61,92-100; lines 257-330", gym_hybrid_sha)),
    }
    rows.append(InstanceExtraction("Gym Hybrid TRM/LDT", hybrid_cells))
    return tuple(rows)
