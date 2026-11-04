"""CCA-11A: deterministic CPU arithmetic; no checkpoint or semantic model used."""
from __future__ import annotations

import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research_gym.core.cca_recoverability_threshold import RecoveryEnvelope


def main():
    env = RecoveryEnvelope(F(1, 2), 1, F(1, 10), 1)
    results = []
    for noise in (F(39, 100), F(2, 5), F(41, 100), F(13, 20)):
        path = env.envelope_path(1, [noise] * 16)
        results.append({
            "noise": str(noise),
            "uniform_certificate": env.uniform_viability_certificate(1, noise),
            "first_envelope_exit": env.first_possible_exit(path),
            "final_envelope": str(path[-1]),
        })
    burst = RecoveryEnvelope(F(1, 2), 1, 0, 1)
    path = burst.envelope_path(0, [F(3, 4), 0, 0, 0])
    artifact = {
        "study": "CCA-11A exact arithmetic toy envelope, no neural checkpoint",
        "threshold": str(env.critical_disturbance()),
        "runs": results,
        "burst": {"uniform_threshold": str(burst.critical_disturbance()),
                  "path": [str(v) for v in path],
                  "first_envelope_exit": burst.first_possible_exit(path)},
    }
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return artifact


if __name__ == "__main__":
    main()
