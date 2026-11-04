/-! Dual-LDT-02: a host-certified temporal intervention shield.

Winning states come from an independently verified game; these propositions
prove the interface between the immediate Beast filter and that game.
-/
universe u v
namespace DualLDT

def InterventionShield {A : Type u} {State : Type v}
    (beastAllowed : A → Prop) (successor : A → State → Prop)
    (hostCanRecover : State → Prop) (a : A) : Prop :=
  beastAllowed a ∧ ∀ s, successor a s → hostCanRecover s

theorem shield_refines_beast {A : Type u} {State : Type v}
    (beastAllowed : A → Prop) (successor : A → State → Prop)
    (hostCanRecover : State → Prop) :
    ∀ a, InterventionShield beastAllowed successor hostCanRecover a →
      beastAllowed a := by
  intro a h
  exact h.1

theorem every_possible_successor_remains_correctable
    {A : Type u} {State : Type v}
    (beastAllowed : A → Prop) (successor : A → State → Prop)
    (hostCanRecover : State → Prop) :
    ∀ a s, InterventionShield beastAllowed successor hostCanRecover a →
      successor a s → hostCanRecover s := by
  intro a s h ha
  exact h.2 s ha

theorem temporal_authority_tightening
    {A : Type u} {State : Type v}
    (oldAllowed newAllowed : A → Prop)
    (hsubset : ∀ a, newAllowed a → oldAllowed a)
    (successor : A → State → Prop)
    (hostCanRecover : State → Prop) :
    ∀ a, InterventionShield newAllowed successor hostCanRecover a →
      InterventionShield oldAllowed successor hostCanRecover a := by
  intro a h
  exact ⟨hsubset a h.1, h.2⟩

end DualLDT
