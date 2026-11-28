# GYM-02 full campaign: protocol deviations

1. **12:20 MPST, CPU reduction at the user's request ("free up 4 cpus").**
   - Two leftover subagent processes were stopped: an unbounded `distill` CLI run and a diagnostic
     script. Their BLAS threads had been using about 34 cores.
   - The campaign queue's process tree (queue, race, workers and all later stages, which inherit the
     setting) was pinned to CPU cores 17–19. The CI-diagnostic replay and the live Inky clients were
     pinned to core 16.
   - Measured campaign usage afterwards was about 0.8 cores, because every core was saturated by other
     work.
   - Stages run slower than the preregistered stage caps assumed, so later stages may be cut by the
     8-hour deadline. Any cut stage is reported as unfinished, never as negative.
2. **CI artifacts.** The GYM-02 diagnostic artifacts could not be downloaded: anonymous artifact
   download returns HTTP 401, and token use is reserved. The same CI commands were re-run locally into
   `experiments/trm_skill_gym_02/ci_replay_local/`.
3. **14:58 MPST, stage reorder.** The CPU reduction made it unlikely that every stage would fit before
   the 19:49 deadline. The pending stages were reordered to S6 distillation and S5 router (cheap), then
   S2 BPD, S4 research, S8 shuffle audit and S7 Inky measurement. No stage parameters changed.
4. **Race coverage.** The 3.0 h calibration cap froze five skills; stopping did not start and is
   unfinished. The two research archive configurations were cut by the deadline and are excluded as
   incomplete.
