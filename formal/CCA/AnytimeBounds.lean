/-! CCA-09: checked incumbent plus bounded unseen actions.

Soundness is conditional on the host proving the upper bound for remaining
actions, and on trusted feasibility and utility checks for evaluated actions.
-/
namespace CCA
universe u v

theorem anytime_upper_sound
    {Action : Type u} {Value : Type v} [Preorder Value]
    (utility : Action → Value)
    (feasible checked : Action → Prop)
    (incumbent : Action) (upper : Value)
    (hchecked : ∀ a, checked a → feasible a → utility a ≤ utility incumbent)
    (hremaining : ∀ a, ¬ checked a → feasible a → utility a ≤ upper)
    (hincumbent : utility incumbent ≤ upper) :
    ∀ a, feasible a → utility a ≤ upper := by
  intro a ha
  by_cases h : checked a
  · exact le_trans (hchecked a h ha) hincumbent
  · exact hremaining a h ha

theorem anytime_target_sound
    {Action : Type u} {Value : Type v} [Preorder Value]
    (utility : Action → Value)
    (feasible checked : Action → Prop)
    (incumbent : Action) (upper target : Value)
    (hchecked : ∀ a, checked a → feasible a → utility a ≤ utility incumbent)
    (hremaining : ∀ a, ¬ checked a → feasible a → utility a ≤ upper)
    (hincumbent : utility incumbent ≤ upper)
    (hgap : upper ≤ target) :
    ∀ a, feasible a → utility a ≤ target := by
  intro a ha
  exact le_trans (anytime_upper_sound utility feasible checked incumbent upper
    hchecked hremaining hincumbent a ha) hgap
end CCA
