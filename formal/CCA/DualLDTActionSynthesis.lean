/-! Dual-LDT-04: sound plan expansion and bounded preference certificates.

Mathematical model: an environment owns the allowed plan predicate and robust
successor relation. Preference is an arbitrary fixed score. Synthesis ranks
within certified plans, with guard prices already folded into the ranking.
-/
universe u v

namespace DualLDT

theorem certified_synthesis_preserves_intervention
    {Plan : Type u} {State : Type v}
    (authorized : Plan → Prop)
    (successor : Plan → State → Prop)
    (recoverable : State → Prop)
    (chosen : Plan)
    (hcertificate : authorized chosen ∧
      ∀ state, successor chosen state → recoverable state) :
    authorized chosen ∧ ∀ state, successor chosen state → recoverable state := by
  exact hcertificate

theorem preference_sorter_cannot_enlarge_authority
    {Plan : Type u}
    (authorized selected : Plan → Prop)
    (hsubset : ∀ a, selected a → authorized a) :
    ∀ a, selected a → authorized a := by
  exact hsubset

theorem certified_expansion_weakly_improves_maximum
    {Plan : Type u} (score : Plan → Nat)
    (oldSafe newSafe : Plan → Prop)
    (oldChoice newChoice : Plan)
    (hsubset : ∀ p, oldSafe p → newSafe p)
    (hOld : oldSafe oldChoice)
    (hNewMax : ∀ p, newSafe p → score p ≤ score newChoice) :
    score oldChoice ≤ score newChoice := by
  exact hNewMax oldChoice (hsubset oldChoice hOld)

theorem witness_establishes_preference_lower_bound
    {Plan : Type u} (score : Plan → Nat)
    (witness chosen : Plan) (guaranteed : Nat)
    (hWitness : guaranteed ≤ score witness)
    (hChosen : score witness ≤ score chosen) :
    guaranteed ≤ score chosen := by
  exact Nat.le_trans hWitness hChosen

end DualLDT
