/-! Dual-LDT-03: three-step future-intervention invariance.

The host's recoverable region and transition correctness are assumptions.
The theorem covers all possible successors of each permitted actor action.
-/
universe u v
namespace DualLDT

theorem three_actor_steps_preserve_host_recovery
    {State : Type u} {Action : Type v}
    (recoverable : State → Prop)
    (transition : State → Action → State → Prop)
    (choose : State → Action)
    (preserves : ∀ s, recoverable s →
      ∀ next, transition s (choose s) next → recoverable next)
    (s0 s1 s2 s3 : State)
    (h0 : recoverable s0)
    (h01 : transition s0 (choose s0) s1)
    (h12 : transition s1 (choose s1) s2)
    (h23 : transition s2 (choose s2) s3) :
    recoverable s3 := by
  have h1 := preserves s0 h0 s1 h01
  have h2 := preserves s1 h1 s2 h12
  exact preserves s2 h2 s3 h23

theorem authenticated_halt_preempts_actor
    {Action : Type u}
    (actorAllowed : Action → Prop)
    (haltReceived : Prop)
    (priority : haltReceived → ∀ a, ¬ actorAllowed a)
    (halt : haltReceived) :
    ∀ a, ¬ actorAllowed a := by
  exact priority halt

end DualLDT
