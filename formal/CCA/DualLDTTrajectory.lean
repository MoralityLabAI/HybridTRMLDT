import Std.Tactic

/-! Dual-LDT-06: uniform-error regret and finite shielded trajectories.
Utility is represented in integer units (experiment denominator 64).
No neural accuracy hypothesis is proved here; it must be supplied independently.
-/
universe u v
namespace DualLDTTrajectory

theorem uniform_value_error_regret
    {Plan : Type u} (value estimate : Plan → Int)
    (chosen reference : Plan) (epsilon : Int)
    (h_reference : value reference ≤ estimate reference + epsilon)
    (h_chosen : estimate chosen ≤ value chosen + epsilon)
    (h_max : estimate reference ≤ estimate chosen) :
    value reference - value chosen ≤ 2 * epsilon := by
  omega

inductive ShieldedPath {State : Type u} {Action : Type v}
    (allowed : State → Action → Prop)
    (transition : State → Action → State → Prop) :
    List Action → State → State → Prop where
  | empty (s : State) : ShieldedPath allowed transition [] s s
  | step {a : Action} {rest : List Action} {s m t : State}
      (ha : allowed s a) (ht : transition s a m)
      (tail : ShieldedPath allowed transition rest m t) :
      ShieldedPath allowed transition (a :: rest) s t

theorem finite_path_preserves_host_recovery
    {State : Type u} {Action : Type v}
    (allowed : State → Action → Prop)
    (transition : State → Action → State → Prop)
    (recoverable : State → Prop)
    (preserves : ∀ s a t, recoverable s → allowed s a →
      transition s a t → recoverable t)
    {steps : List Action} {s t : State}
    (execution : ShieldedPath allowed transition steps s t) :
    recoverable s → recoverable t := by
  induction execution with
  | empty s => exact fun hs => hs
  | step ha ht tail ih =>
      intro hs
      exact ih (preserves _ _ _ hs ha ht)

#print axioms uniform_value_error_regret
#print axioms finite_path_preserves_host_recovery
end DualLDTTrajectory
