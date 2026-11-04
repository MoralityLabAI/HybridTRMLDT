/-! Dual-LDT-01: the Beast as an interior operator on action predicates.

Theorems concern the fixed host-authorized safe predicate. They do not
authenticate policy updates or verify semantic tag extraction.
-/
universe u
namespace DualLDT

def Beast {A : Type u} (safe proposed : A → Prop) (a : A) : Prop :=
  proposed a ∧ safe a

theorem contractive {A : Type u} (safe proposed : A → Prop) :
    ∀ a, Beast safe proposed a → proposed a := by
  intro a h
  exact h.1

theorem authorization {A : Type u} (safe proposed : A → Prop) :
    ∀ a, Beast safe proposed a → safe a := by
  intro a h
  exact h.2

theorem monotone {A : Type u} (safe x y : A → Prop)
    (hsub : ∀ a, x a → y a) :
    ∀ a, Beast safe x a → Beast safe y a := by
  intro a h
  exact ⟨hsub a h.1, h.2⟩

theorem idempotent {A : Type u} (safe x : A → Prop) :
    ∀ a, Beast safe (Beast safe x) a ↔ Beast safe x a := by
  intro a
  constructor
  · intro h
    exact ⟨h.1.1, h.2⟩
  · intro h
    exact ⟨h, h.2⟩

theorem meet_preserving {A : Type u} (safe x y : A → Prop) :
    ∀ a, Beast safe (fun b => x b ∧ y b) a ↔
      Beast safe x a ∧ Beast safe y a := by
  intro a
  constructor
  · intro h
    exact ⟨⟨h.1.1, h.2⟩, ⟨h.1.2, h.2⟩⟩
  · intro h
    exact ⟨⟨h.1.1, h.2.1⟩, h.1.2⟩

theorem corrected_authority {A : Type u} (oldSafe newSafe x : A → Prop)
    (hnew : ∀ a, newSafe a → oldSafe a) :
    ∀ a, Beast newSafe (Beast oldSafe x) a ↔ Beast newSafe x a := by
  intro a
  constructor
  · intro h
    exact ⟨h.1.1, h.2⟩
  · intro h
    exact ⟨⟨h.1, hnew a h.2⟩, h.2⟩

theorem no_permission_from_persona {A : Type u}
    (safe : A → Prop) (jinnRanking proposed : A → Prop)
    (hchoice : ∀ a, jinnRanking a → Beast safe proposed a) :
    ∀ a, jinnRanking a → safe a := by
  intro a ha
  exact (hchoice a ha).2

end DualLDT
