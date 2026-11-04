/-!
Dual-LDT-05: universal-branch composition under changing human authority.

The host-supplied relation P is executed before the correction. Relation Q is
executed under the *new* authority predicate, evaluated on every successor.
Program certificate soundness requires every reachable intermediate state to
satisfy the second-step authorization and every resulting state to remain in W.
-/ 
universe u
namespace DualLDT

def Preserves {S : Type u}
    (W : S → Prop) (R : S → S → Prop) : Prop :=
  ∀ s t, W s → R s t → W t

def Compose {S : Type u}
    (P Q : S → S → Prop) (s t : S) : Prop :=
  ∃ m, P s m ∧ Q m t

theorem invariant_preserved_by_composition
    {S : Type u}
    (W : S → Prop) (P Q : S → S → Prop)
    (hP : Preserves W P) (hQ : Preserves W Q) :
    Preserves W (Compose P Q) := by
  intro s t hs hpq
  cases hpq with
  | intro m hpath =>
    exact hQ m t (hP s m hs hpath.1) hpath.2

def CertifiedTwoStep {S : Type u}
    (W : S → Prop) (oldAuth newAuth : S → Prop)
    (P Q : S → S → Prop) (start : S) : Prop :=
  W start ∧ oldAuth start ∧
  ∀ mid, P start mid →
    W mid ∧ newAuth mid ∧
    (∀ final, Q mid final → W final)

theorem checked_program_preserves_intervention
    {S : Type u}
    (W : S → Prop) (oldAuth newAuth : S → Prop)
    (P Q : S → S → Prop) (start mid final : S)
    (hcert : CertifiedTwoStep W oldAuth newAuth P Q start)
    (hfirst : P start mid) (hsecond : Q mid final) :
    W final := by
  exact (hcert.2.2 mid hfirst).2.2 final hsecond

theorem authorization_rechecked_after_correction
    {S : Type u}
    (W : S → Prop) (oldAuth newAuth : S → Prop)
    (P Q : S → S → Prop) (start mid : S)
    (hcert : CertifiedTwoStep W oldAuth newAuth P Q start)
    (hfirst : P start mid) :
    newAuth mid := by
  exact (hcert.2.2 mid hfirst).2.1

theorem tighter_intermediate_authority_cannot_create_programs
    {S : Type u}
    (W : S → Prop) (oldAuth newAuth strictAuth : S → Prop)
    (P Q : S → S → Prop) (start : S)
    (htight : ∀ s, strictAuth s → newAuth s)
    (hcert : CertifiedTwoStep W oldAuth strictAuth P Q start) :
    CertifiedTwoStep W oldAuth newAuth P Q start := by
  refine ⟨hcert.1, hcert.2.1, ?_⟩
  intro mid hmid
  have h := hcert.2.2 mid hmid
  exact ⟨h.1, htight mid h.2.1, h.2.2⟩

end DualLDT
