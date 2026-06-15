# Repository Sync and MoralityLabAI Org Status

Date: 2026-06-15

## Summary

- `metta-storyworld` is present locally at `C:\projects\metta-storyworld\metta-storyworld`.
- The local `metta-storyworld` checkout is newer than `origin/main`: `main` is 26 commits ahead of `origin/main`.
- The latest committed `metta-storyworld` change is `2a79170` from 2026-05-08: `Add VPD keygate village seed`.
- The local `metta-storyworld` tree also has many uncommitted changes, including research notes, traces, scenario files, agent files, and tests.
- `TRMStoryworld` is already configured against `https://github.com/MoralityLabAI/TRMStoryworld.git`.
- `HybridTRMLDT`, `Hermes-Skills`, and `VPD` have local MoralityLabAI remotes configured, but the corresponding org repos are not currently reachable.

## Local Paths

| Project | Path | Branch | Remote State |
| --- | --- | --- | --- |
| HybridTRMLDT | `C:\projects\HybridTRMLDT\ldt_trm_research_gym_v0_0_2\ldt_trm_research_gym` | `master` | `origin` set to `https://github.com/MoralityLabAI/HybridTRMLDT.git`; remote not found |
| metta-storyworld | `C:\projects\metta-storyworld\metta-storyworld` | `main` | `origin` set to `https://github.com/MoralityLabAI/metta-storyworld.git`; local branch ahead by 26 commits |
| Hermes-Skills | `C:\projects\Hermes-Skills\Hermes Skills` | `main` | `origin` set to `https://github.com/patrickdugan/Hermes-Skills.git`; `moralitylabai` set to `https://github.com/MoralityLabAI/Hermes-Skills.git`; org remote not found |
| VPD | `C:\projects\VPD` | `rust-accel-scoring` | `origin` set to `https://github.com/patrickdugan/param-decomp.git`; `upstream` set to `https://github.com/goodfire-ai/param-decomp.git`; `moralitylabai` set to `https://github.com/MoralityLabAI/VPD.git`; org remote not found |

## Actions Taken

- Fetched `metta-storyworld` remotes with prune enabled.
- Confirmed `metta-storyworld` does not need a pull from `origin/main`; the local checkout is ahead of the org remote.
- Did not merge, rebase, stash, or commit inside dirty external repositories.
- Added local MoralityLabAI remotes:
  - `HybridTRMLDT`: `origin -> https://github.com/MoralityLabAI/HybridTRMLDT.git`
  - `Hermes-Skills`: `moralitylabai -> https://github.com/MoralityLabAI/Hermes-Skills.git`
  - `VPD`: `moralitylabai -> https://github.com/MoralityLabAI/VPD.git`
- Checked org repository availability with `git ls-remote`; `HybridTRMLDT`, `Hermes-Skills`, `VPD`, and `param-decomp` under `MoralityLabAI` were not found or not accessible anonymously.
- Confirmed no GitHub CLI is installed and no `GITHUB_TOKEN`, `GH_TOKEN`, or `GIT_TOKEN` environment variable is present.

## Blockers

- Creating repositories in the `MoralityLabAI` organization requires authenticated GitHub access.
- `gh` is not installed on PATH.
- No GitHub token is available in the shell environment.
- `metta-storyworld`, `Hermes-Skills`, `TRMStoryworld`, and `VPD` all have dirty working trees; pushing them as-is would not include uncommitted local work.

## Recommended Next Safe Steps

1. Create the missing GitHub org repositories:
   - `MoralityLabAI/HybridTRMLDT`
   - `MoralityLabAI/Hermes-Skills`
   - `MoralityLabAI/VPD`
2. Push only committed history first:
   - HybridTRMLDT: `git push -u origin master`
   - Hermes-Skills: `git push -u moralitylabai main`
   - VPD: `git push -u moralitylabai rust-accel-scoring`
3. Separately review and commit or stash dirty work in `metta-storyworld`, `Hermes-Skills`, `TRMStoryworld`, and `VPD` before pushing any uncommitted updates.
4. Push `metta-storyworld` to `origin/main` after reviewing the dirty tree, since local committed history is already ahead of the org remote.
