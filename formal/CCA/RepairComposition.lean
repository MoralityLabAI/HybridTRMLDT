/-! CCA-06: core proofs of commuting repairs and sound abstraction. -/
universe u v
namespace CCA

def IsRetractionOn {α : Type u} (R : α → α) (C : α → Prop) : Prop :=
  (∀ x, C (R x)) ∧ (∀ x, C x → R x = x)

theorem commuting_retractions
    {α : Type u} {R Q : α → α} {C D : α → Prop}
    (hR : IsRetractionOn R C)
    (hQ : IsRetractionOn Q D)
    (hcomm : ∀ x, R (Q x) = Q (R x)) :
    IsRetractionOn (fun x => R (Q x)) (fun x => C x ∧ D x) := by
  constructor
  · intro x
    constructor
    · exact hR.1 (Q x)
    · rw [hcomm x]
      exact hQ.1 (R x)
  · intro x hx
    rw [hQ.2 x hx.2, hR.2 x hx.1]

theorem commuting_retractions_idempotent
    {α : Type u} {R Q : α → α} {C D : α → Prop}
    (hR : IsRetractionOn R C)
    (hQ : IsRetractionOn Q D)
    (hcomm : ∀ x, R (Q x) = Q (R x)) :
    ∀ x, R (Q (R (Q x))) = R (Q x) := by
  intro x
  have h := commuting_retractions hR hQ hcomm
  exact h.2 (R (Q x)) (h.1 x)

def UniformlyAllowed {σ : Type u} {α : Type v}
    (possible : σ → Prop) (permitted : σ → α → Prop) (a : α) : Prop :=
  ∀ s, possible s → permitted s a

theorem authorized_action_sound
    {σ : Type u} {α : Type v}
    {possible : σ → Prop} {permitted : σ → α → Prop}
    {actual : σ} {a : α}
    (hactual : possible actual)
    (ha : UniformlyAllowed possible permitted a) :
    permitted actual a := by
  exact ha actual hactual

theorem refinement_preserves_permissions
    {σ : Type u} {α : Type v}
    {coarse fine : σ → Prop} {permitted : σ → α → Prop}
    (hsub : ∀ s, fine s → coarse s) :
    ∀ a, UniformlyAllowed coarse permitted a →
      UniformlyAllowed fine permitted a := by
  intro a ha s hs
  exact ha s (hsub s hs)

end CCA
