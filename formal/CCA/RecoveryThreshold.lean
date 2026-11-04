/-!
CCA-11 discrete scaled-integer recovery invariant.

The induction proves the boundary-step condition yields an invariant viable
region. A real/rational physical error bound requires sound discretization
before this particular theorem is applicable.
-/
namespace CCA

def worstCaseEnvelope (rho gain noise residual initial : Nat) : Nat → Nat
  | 0 => initial
  | t + 1 =>
      rho * worstCaseEnvelope rho gain noise residual initial t + gain * noise + residual

theorem recovery_step_bounded
    (rho gain noise residual margin e : Nat)
    (hboundary : rho * margin + gain * noise + residual ≤ margin)
    (he : e ≤ margin) :
    rho * e + gain * noise + residual ≤ margin := by
  have hmul : rho * e ≤ rho * margin := Nat.mul_le_mul_left rho he
  have hadd : rho * e + gain * noise ≤ rho * margin + gain * noise :=
    Nat.add_le_add_right hmul (gain * noise)
  have hres : rho * e + gain * noise + residual ≤
      rho * margin + gain * noise + residual :=
    Nat.add_le_add_right hadd residual
  exact Nat.le_trans hres hboundary

theorem recovery_envelope_bounded
    (rho gain noise residual initial margin : Nat)
    (hboundary : rho * margin + gain * noise + residual ≤ margin)
    (hinit : initial ≤ margin) :
    ∀ t, worstCaseEnvelope rho gain noise residual initial t ≤ margin := by
  intro t
  induction t with
  | zero => exact hinit
  | succ t ih =>
      exact recovery_step_bounded rho gain noise residual margin
        (worstCaseEnvelope rho gain noise residual initial t) hboundary ih

end CCA
