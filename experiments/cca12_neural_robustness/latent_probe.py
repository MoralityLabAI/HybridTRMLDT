"""CCA-12B: optional hash-verified 48D final-hidden perturbation study.

This requires the six original .pt checkpoints, which CCA-05 stores outside Git.
CPU inference is sufficient. It refuses altered model files and tape mismatch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
CCA05 = REPO / "experiments" / "cca05_certified_trm_repair"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CCA05 / "src"))

import torch  # noqa: E402
import semantics as M  # noqa: E402
from tape import input_key, read_gz  # noqa: E402
from research_gym.core.hybrid import CandidateState  # noqa: E402
from research_gym.neural.trm import TRMProposer  # noqa: E402
from research_gym.core.cca_neural_margins import (  # noqa: E402
    anchored_hidden_l2_radius, winner,
)

REG = json.loads((CCA05 / "registration.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((CCA05 / "outputs" / "manifest_live.json").read_text(encoding="utf-8"))


def state_dict_digest(sd):
    h = hashlib.sha256()
    for name, val in sorted(sd.items()):
        h.update(name.encode())
        h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_verified(tag, ckdir):
    item = REG["checkpoints"][tag]
    path = ckdir / item["file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["file_sha256"]:
        raise RuntimeError(f"{tag}: original file SHA256 mismatch")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_digest(sd) != item["state_dict_sha256"]:
        raise RuntimeError(f"{tag}: original state dict SHA256 mismatch")
    model = TRMProposer(latent_dim=48, recurrence_steps=4)
    model.load_state_dict(sd)
    model.eval()
    return model


def run(ckdir, out, tags):
    torch.set_num_threads(1)
    reports = {}
    for tag in tags:
        model = load_verified(tag, ckdir)
        tape_path = CCA05 / "outputs" / "tapes" / f"{tag}.json.gz"
        if hashlib.sha256(tape_path.read_bytes()).hexdigest() != MANIFEST["tapes/" + f"{tag}.json.gz"]:
            raise RuntimeError("Tape digest mismatch")
        tape = read_gz(tape_path)
        row_weights = model.action_head.weight.detach().cpu().tolist()
        seed = REG["checkpoints"][tag]["seed"]
        # Separate from proposer_train; pre-declared held-out buckets 8,9.
        sampled = M.sampled_regions(seed)["heldout_high_trust"]
        recorded = []
        parity_errors = 0
        guard_violations = 0
        for scenario, state in sampled:
            allowed = M.certified_set(scenario, state, M.HORIZON)
            features = model.features_from(scenario, CandidateState(
                {"action": frozenset(M.ACTIONS)}), state)
            with torch.no_grad():
                logits = tuple(map(float, model(features).action_logits[0].tolist()))
            saved = tuple(tape["full_mask_passes"][input_key(scenario, state)][3])
            max_diff = max(abs(a-b) for a,b in zip(saved, logits))
            if max_diff > 2e-5:
                parity_errors += 1
                continue
            r = anchored_hidden_l2_radius(saved, row_weights, M.ACTIONS, allowed)
            if r is not None and math.isfinite(r) and r > 0:
                idx = M.ACTIONS.index(winner(saved, M.ACTIONS))
                unsafe = [u for u,a in enumerate(M.ACTIONS) if a not in allowed]
                competitor = min(unsafe, key=lambda u: (
                    (saved[idx]-saved[u])/
                    math.sqrt(sum((a-b)**2 for a,b in zip(row_weights[idx], row_weights[u])))
                    if any(a!=b for a,b in zip(row_weights[idx], row_weights[u]))
                    else math.inf))
                difference = [a-b for a,b in zip(row_weights[idx], row_weights[competitor])]
                norm = math.sqrt(sum(d*d for d in difference))
                if norm:
                    # Pick a boundary-directed perturbation strictly inside the proven radius.
                    delta = [-0.99*r*d/norm for d in difference]
                    new_logits = [z+sum(w_i*d_i for w_i,d_i in zip(row,delta))
                                  for z,row in zip(saved,row_weights)]
                    if winner(new_logits, M.ACTIONS) not in allowed:
                        guard_violations += 1
            recorded.append({"scenario": scenario, "state": list(M.st(state)),
                             "radius": r, "max_tape_parity_error": max_diff,
                             "baseline_safe": winner(saved,M.ACTIONS) in allowed})
        if parity_errors or guard_violations:
            raise RuntimeError(f"{tag}: {parity_errors} tape parity errors; "
                               f"{guard_violations} certified-radius failures")
        finite_positive = sorted(v["radius"] for v in recorded
                                 if v["radius"] is not None
                                 and math.isfinite(v["radius"]) and v["radius"] > 0)
        reports[tag] = {
            "verified_checkpoint_sha256": REG["checkpoints"][tag]["file_sha256"],
            "tape_parity_tested": len(sampled),
            "positive_finite_radius_count": len(finite_positive),
            "median_positive_radius": (finite_positive[len(finite_positive)//2]
                                       if finite_positive else None),
            "guard_violations": guard_violations,
        }
        print(f'{tag}: parity={len(sampled)}, positive_finite_radii='
              f'{len(finite_positive)}, guard_violations={guard_violations}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"study":"CCA-12B original checkpoints, held-out "
                               "48D final-latent additive perturbation",
                               "reports":reports},indent=2,sort_keys=True)+"\n")
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("cca12_latent_private.json"))
    parser.add_argument("--tags", nargs="*", default=list(REG["checkpoints"]))
    args = parser.parse_args()
    run(args.checkpoint_dir, args.out, args.tags)


if __name__ == "__main__":
    main()
