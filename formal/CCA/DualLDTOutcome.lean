import Std.Tactic

/-! Post-run Dual-LDT-06 mathematics: exact outcome dominance.

The host supplies already-certified feasible plans and exact outcome/cost
numerators. The following proof concerns nonnegative integer persona weights.
Scaling a common denominator extends the mathematical argument to nonnegative
rational weights. No neural approximation premise is supplied by these lemmas.
-/
namespace DualLDTOutcome

def scale : Nat → Int → Int
  | 0, _ => 0
  | Nat.succ n, x => x + scale n x

theorem scale_mono (n : Nat) {a b : Int} (h : a ≤ b) :
    scale n a ≤ scale n b := by
  induction n with
  | zero => simp [scale]
  | succ n ih =>
      simp only [scale]
      omega

structure Outcome where
  f0 : Int
  f1 : Int
  f2 : Int
  f3 : Int
  cost : Int

def Dominates (a b : Outcome) : Prop :=
  b.f0 ≤ a.f0 ∧ b.f1 ≤ a.f1 ∧ b.f2 ≤ a.f2 ∧
  b.f3 ≤ a.f3 ∧ a.cost ≤ b.cost

def value (w0 w1 w2 w3 : Nat) (a : Outcome) : Int :=
  scale w0 a.f0 + scale w1 a.f1 + scale w2 a.f2 + scale w3 a.f3 - a.cost

theorem dominance_preserves_all_personas
    (w0 w1 w2 w3 : Nat) (a b : Outcome) (h : Dominates a b) :
    value w0 w1 w2 w3 b ≤ value w0 w1 w2 w3 a := by
  have h0 := scale_mono w0 h.1
  have h1 := scale_mono w1 h.2.1
  have h2 := scale_mono w2 h.2.2.1
  have h3 := scale_mono w3 h.2.2.2.1
  have hc := h.2.2.2.2
  simp only [value]
  omega

universe u

theorem coverage_preserves_optimum
    {Plan : Type u} (score : Plan → Int)
    (feasible retained : Plan → Prop) (chosen : Plan)
    (cover : ∀ p, feasible p → ∃ q, retained q ∧ score p ≤ score q)
    (best : ∀ q, retained q → score q ≤ score chosen) :
    ∀ p, feasible p → score p ≤ score chosen := by
  intro p hp
  cases cover p hp with
  | intro q hq =>
      exact Int.le_trans hq.2 (best q hq.1)

#print axioms dominance_preserves_all_personas
#print axioms coverage_preserves_optimum
end DualLDTOutcome
