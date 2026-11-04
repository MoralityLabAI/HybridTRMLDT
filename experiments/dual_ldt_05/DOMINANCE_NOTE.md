# Dual-LDT-05 diagnostic: Why can new certified plans leave behavior unchanged?

This is a **post-hoc mathematical diagnosis**, introduced only after the registered 64-world report replay showed zero new BUILD actions selected. It cannot be used to change the registered generator, hyperparameters, head weights or outcome verdict.

Let B be the BASE certified program set, and let E be the added BUILD programs under COMPOSED. For each program a, define fixed adjusted persona utility v(a)=log P_head(train|host tags(a))-0.10*credits(a). Let

    M = max_{a in E} v(a) - max_{b in B} v(b)

where an empty E produces an unassessable margin.

**Exact argmax-change condition:** if M>0, a newly added program must become the argmax in COMPOSED. If M<0, the previously selected BASE program remains preferred. At M=0, tie order determines the result. The proof follows from maximization over B union E. The diagnostic recomputes every program's host-certified score and preserves this margin per world and persona.

The earlier Dual-LDT-05 result found an average of roughly four newly certified BUILD programs per world and zero BUILD selections among four fixed Jinn heads. The new diagnostic records the maximum score deficit, including whether any candidate ties the baseline winner, while leaving the study's already frozen endpoint unchanged.

A reduced conditional KL preference tax can coexist with no change in argmax choice, since it depends on added probability mass rather than the highest adjusted score. This distinction is central to interpreting the experiment. No neural training and no new split are introduced.
