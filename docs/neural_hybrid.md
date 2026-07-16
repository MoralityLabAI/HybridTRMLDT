# Optional Neural Hybrid

## Scope

`research_gym/neural/` provides a small recurrent proposer, deterministic storyworld rollout helpers, and linear
probe tools. PyTorch is an optional dependency. The lattice, membrane, environments, and existing scripts remain
importable without third-party packages.

Install the optional component with:

```powershell
pip install -e ".[neural]"
```

## TRM Proposer

`TRMProposer` encodes:

- a target predicate (`secret_ending` or `moral_optimization`);
- normalized story state;
- the current action candidate mask.

The default model uses a 256-dimensional latent and six shared GRU recurrence steps. Four heads emit action,
`HybridMode`, conflict, and claimed `SoundnessType`. `propose(..., return_latents=True)` returns every recurrent
latent and emits a dependency-free `LatticeProposal` directly.

The gaming benchmark uses a smaller registered model for bounded local execution. The class defaults remain the
upstream contract defaults.

## Rollout

`rollout_examples` implements:

```text
propose -> verify/certify -> select proposal or fallback -> retain typed receipt
```

It is deterministic given the model checkpoint, examples, policy, and seed. Story examples have group IDs formed
from a deterministic hash of the start state. Training, probe calibration, and evaluation use contiguous trust
regions with zero group overlap.

The state-conditioned fallback is separately implemented. It uses exact reachability for the secret-ending
scenario and a fixed safe-action priority for moral optimization; it is not the utility oracle.

## Probe Protocol

`fit_grouped_probe` provides:

- group-disjoint training and evaluation splits;
- binary AUROC;
- a within-start-state shuffled-label floor;
- a threshold fitted only on probe-calibration groups;
- explicit probe weights for cosine drift measurement.

`matched_control_directions` emits multiple matched-norm random and orthogonal controls. The benchmark records
probe-direction intervention effects over three seeds. A probe remains endogenous to the proposer and can age or
be optimized against even when no gradient crosses the verifier.

## Dependency Check

Core import without site packages:

```powershell
python -S -c "import sys; sys.path.insert(0, '.'); import research_gym.core; print('core import passed')"
```

Neural tests skip when PyTorch is absent. Default membrane tests do not import the neural package.

