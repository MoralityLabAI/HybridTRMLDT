"""Hash-verified TRM checkpoints and the recording proposer (requires torch)."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from research_gym.core.hybrid import CandidateState
from research_gym.core.typed_soundness import SoundnessType
from research_gym.neural.trm import SOUNDNESS_TYPES, TRMProposer

import semantics as M
from tape import input_key, query_key

ROOT = Path(__file__).resolve().parents[1]
REG = json.loads((ROOT / "registration.json").read_text(encoding="utf-8"))
EXTENDED_PASSES = 8


def state_dict_sha256(sd) -> str:
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_checkpoint(tag: str) -> tuple[TRMProposer, dict]:
    """Load trm_{kind}_s{seed}.pt; refuse unless both hashes equal the registration."""
    ck = REG["checkpoints"][tag]
    path = Path(REG["checkpoint_dir"]) / ck["file"]
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != ck["file_sha256"]:
        raise RuntimeError(f"{tag}: file hash {file_sha} != registered {ck['file_sha256']}")
    sd = torch.load(path, map_location="cpu")
    sd_sha = state_dict_sha256(sd)
    if sd_sha != ck["state_dict_sha256"]:
        raise RuntimeError(f"{tag}: state-dict hash mismatch")
    model = TRMProposer(latent_dim=48, recurrence_steps=4)
    model.load_state_dict(sd)
    model.eval()
    return model, {"file": str(path), "file_sha256": file_sha, "state_dict_sha256": sd_sha,
                   "bytes": len(raw), "parameters": sum(p.numel() for p in model.parameters())}


class LiveTRM:
    """Single-row forwards (as TRMProposer.propose runs them), cached and recorded."""

    def __init__(self, model: TRMProposer) -> None:
        self.model = model
        self.full: dict[str, list[list[float]]] = {}
        self.masked: dict[str, list[float]] = {}
        self.claims: dict[str, float] = {}
        self.forwards = 0

    def _forward(self, scenario, state, mask, passes):
        feats = self.model.features_from(scenario, CandidateState({"action": mask}), state)
        old = self.model.recurrence_steps
        self.model.recurrence_steps = passes
        try:
            with torch.no_grad():
                out = self.model(feats)
        finally:
            self.model.recurrence_steps = old
        self.forwards += 1
        return out

    def final_logits(self, scenario, state, mask):
        if mask == frozenset(M.ACTIONS):
            return tuple(self.pass_logits(scenario, state)[3])
        key = query_key(scenario, state, mask)
        if key not in self.masked:
            out = self._forward(scenario, state, mask, 4)
            self.masked[key] = [float(v) for v in out.action_logits[0]]
        return tuple(self.masked[key])

    def pass_logits(self, scenario, state):
        key = input_key(scenario, state)
        if key not in self.full:
            out = self._forward(scenario, state, frozenset(M.ACTIONS), EXTENDED_PASSES)
            with torch.no_grad():
                self.full[key] = [[float(v) for v in self.model.action_head(out.latents[:, t])[0]]
                                  for t in range(EXTENDED_PASSES)]
            base = self._forward(scenario, state, frozenset(M.ACTIONS), 4)
            if [float(v) for v in base.action_logits[0]] != self.full[key][3]:
                raise AssertionError("pass-4 logits differ between 4- and 8-pass forwards")
            env = SOUNDNESS_TYPES.index(SoundnessType.ENV_SOUND_DEAD)
            unk = SOUNDNESS_TYPES.index(SoundnessType.UNKNOWN)
            pair = torch.softmax(base.provenance_logits[0, [env, unk]], dim=0)
            self.claims[key] = float(pair[0])
        return tuple(tuple(row) for row in self.full[key])

    def tape(self, tag: str, meta: dict) -> dict:
        return {"checkpoint": tag, **meta, "passes_recorded": EXTENDED_PASSES,
                "full_mask_passes": self.full, "masked_final": self.masked,
                "env_claim_score": self.claims}


def reconstructed_claim_threshold(model: TRMProposer, seed: int, kind: str) -> float:
    """X keeps the default 0.5. R's calibrated threshold is not in the state
    dict; _undertrained_base sets it to the 0.9 quantile of env_claim_score over
    proposer-train proposals that fail Verify. Recomputed here from that rule."""
    if kind == "X":
        return 0.5
    live = LiveTRM(model)
    scores = []
    for sc, s in M.sampled_regions(seed)["proposer_train"]:
        live.pass_logits(sc, s)
        a = M.ACTIONS[max(range(5), key=lambda i: (live.full[input_key(sc, s)][3][i], -i))]
        if not M.admissible(sc, s, M.HORIZON, a):
            scores.append(live.claims[input_key(sc, s)])
    ordered = sorted(scores)
    return ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))] if ordered else 0.5


def published_tree_parity(tag: str, live: LiveTRM) -> dict:
    """Agreement of the final-pass action with complete_accounts_v1's exact tree."""
    paper = Path(REG["paper_repo"])
    rel = f"experiments/complete_accounts_v1/outputs/part_b/rules/{tag}_final_pass_trees.metta"
    found = sorted(glob.glob(os.path.expandvars(REG["git_exe_glob"])))
    git = found[-1] if found else "git"
    try:
        text = subprocess.run([git, "-C", str(paper), "show", f"{REG['paper_commit']}:{rel}"],
                              capture_output=True, text=True, check=True).stdout
        blob = subprocess.run([git, "-C", str(paper), "rev-parse", f"{REG['paper_commit']}:{rel}"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}
    leaves = []
    for line in text.splitlines():
        parts = line.strip("()").split()
        if len(parts) < 4 or parts[1] != f"{tag}_action_pass4":
            continue
        lits = [(p[0] == "+", p[1:].split(">=")[0], int(p[1:].split(">=")[1])) for p in parts[4:]]
        leaves.append((parts[3], lits))
    agree = total = 0
    for sc, s in M.full_domain():
        coords = {"scenario": M.SCENARIOS.index(sc), "trust": s.trust, "evidence": s.evidence,
                  "heat": s.heat, "scene": s.scene}
        hits = [lab for lab, lits in leaves if all((coords[c] >= t) == ge for ge, c, t in lits)]
        logits = live.pass_logits(sc, s)[3]
        mine = M.ACTIONS[max(range(5), key=lambda i: (logits[i], -i))]
        total += 1
        agree += len(hits) == 1 and hits[0] == mine
    return {"available": True, "blob": blob, "leaves": len(leaves), "inputs": total,
            "agreement": agree / total}
