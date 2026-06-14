from __future__ import annotations

import argparse
from pathlib import Path


TASKS = {
    "00_agent_protocol.md": """# Agent Protocol\n\nOperate in small, inspectable steps.\n\n1. Read `README.md`, `docs/hybrid_architecture.md`, and the current workorder.\n2. Run the quickstart commands.\n3. Modify at most three files per pass.\n4. Write a short `handoff.md` containing: files changed, commands run, observed failures, next action.\n5. Stop at any checkpoint labeled STOP.\n\nDo not invent citations. Mark missing citations as `CITE_NEEDED`. Do not claim VPD editing exists. Say `VPD-instrumented` or `VPD-guided` unless a concrete feedback mechanism is implemented.\n""",
    "01_lit_clearance.md": """# Workorder 01: Literature Clearance\n\nGoal: produce `docs/lit_clearance.md`.\n\nScope:\n- may/must abstraction for games\n- ATL / GR(1) / controllable predecessor\n- shielding and safe RL\n- neuro-symbolic RL with learned pruning\n- option foreclosure / attainable utility preservation\n- interactive narrative planning\n\nOutput format per item:\n\n```text\nClaim:\nVerdict: CLEAR | PARTIAL | COLLISION | UNKNOWN\nEvidence:\nCitation:\nRisk to project:\nReframe if needed:\n```\n\nSTOP after writing the file.\n""",
    "02_formalization.md": """# Workorder 02: Formalization\n\nGoal: produce `docs/formalization.md` after lit clearance.\n\nRequired sections:\n1. Hybrid LDT/TRM two-state architecture.\n2. Abstract lattice state `a_t` and latent recurrent state `h_t`.\n3. Typed conflict judgments: environment-sound, model-sound, experience-sound.\n4. Projection/certification layer: TRM proposes, lattice contains.\n5. Metta frame classes: execution frames, deduction frames, transition frames.\n6. Storyworld as first gym environment, not the whole contribution.\n\nSTOP after writing the file.\n""",
    "03_experiment_specs.md": """# Workorder 03: Experiment Specs\n\nGoal: produce `docs/experiments.md`.\n\nRungs:\nE0: frame-level alpha/deduction supervision on synthetic frames.\nE1: toy storyworld frame generation and symbolic detector.\nE2: LDT-only candidate/conflict head.\nE3: hybrid LDT/TRM with cross-step latent.\nE4: VPD instrumentation over trained heads.\nE5: Metta code/mechanics frame ingestion.\n\nEach rung needs hypothesis, input frames, baseline, metric, compute, kill criterion.\n""",
    "04_implementation_next.md": """# Workorder 04: Implementation Next\n\nGoal: extend the toy gym without breaking quickstart.\n\nAllowed next steps:\n- Add `research_gym/core/metta_frames.py` with dataclasses only.\n- Add a tiny logistic or MLP baseline using only optional dependencies.\n- Add exact-vs-box abstraction comparison.\n- Add CLI to split train/eval JSONL.\n\nDo not add heavy dependencies by default.\n""",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write small-agent workorders.")
    parser.add_argument("--out", type=Path, default=Path("tasks/generated"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, body in TASKS.items():
        path = args.out / name
        path.write_text(body, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
