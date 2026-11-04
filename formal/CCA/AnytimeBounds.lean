/-! CCA-09: generic verified upper bound for finite anytime search.

The host supplies a transitive utility comparison relation, independently
checked feasible action outcomes, and upper bounds on unseen possibilities.
-/
namespace CCA
universe u v

theorem anytime_upper_sound
    {Action : Type u} {Value : Type v}
    (leq : Value → Value → Prop)
    (htrans : ∀ x y z, leq x y → leq y z → leq x z)
    (utility : Action → Value)
    (feasible checked : Action → Prop)
    (incumbent : Action) (upper : Value)
    (hchecked : ∀ a, checked a → feasible a →
      leq (utility a) (utility incumbent))
    (hremaining : ∀ a, ¬ checked a → feasible a → leq (utility a) upper)
    (hincumbent : leq (utility incumbent) upper) :
    ∀ a, feasible a → leq (utility a) upper := by
  intro a ha
  by_cases h : checked a
  · exact htrans (utility a) (utility incumbent) upper
      (hchecked a h ha) hincumbent
  · exact hremaining a h ha

theorem anytime_target_sound
    {Action : Type u} {Value : Type v}
    (leq : Value → Value → Prop)
    (htrans : ∀ x y z, leq x y → leq y z → leq x z)
    (utility : Action → Value)
    (feasible checked : Action → Prop)
    (incumbent : Action) (upper target : Value)
    (hchecked : ∀ a, checked a → feasible a →
      leq (utility a) (utility incumbent))
    (hremaining : ∀ a, ¬ checked a → feasible a → leq (utility a) upper)
    (hincumbent : leq (utility incumbent) upper)
    (hgap : leq upper target) :
    ∀ a, feasible a → leq (utility a) target := by
  intro a ha
  have hupper := anytime_upper_sound leq htrans utility feasible
    checked incumbent upper hchecked hremaining hincumbent a ha
  exact htrans (utility a) upper target hupper hgap
end CCA
