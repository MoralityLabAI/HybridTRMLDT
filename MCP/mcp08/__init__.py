"""MCP-08: Context economics and autonomous world construction."""
from .fixtures import generate_suite, assert_zero_witness_leakage
from .verifiers import (
    verify_submission,
    verify_clockmakers_republic,
    verify_garden_of_unmeasured_things,
    compute_eta,
    paired_bootstrap_ci,
    evaluate_gates,
)

__all__ = [
    "generate_suite",
    "assert_zero_witness_leakage",
    "verify_submission",
    "verify_clockmakers_republic",
    "verify_garden_of_unmeasured_things",
    "compute_eta",
    "paired_bootstrap_ci",
    "evaluate_gates",
]
