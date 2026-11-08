"""Finite-input event-clock GAE reference. No model, network, or environment effects."""
from dataclasses import dataclass
import math
from typing import Sequence

@dataclass(frozen=True)
class Step:
    reward: float
    value: float
    gamma: float = 1.0
    lam: float = 1.0
    terminal: bool = False

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.reward,self.value,self.gamma,self.lam)):
            raise ValueError('All numeric fields must be finite')
        if not 0 <= self.gamma <= 1 or not 0 <= self.lam <= 1:
            raise ValueError('Discount and trace factors must lie in [0,1]')
        if type(self.terminal) is not bool:
            raise TypeError('terminal must be an explicit bool')

def gae(steps: Sequence[Step], *, next_value: float = 0.0,
        next_advantage: float = 0.0) -> list[float]:
    """Exact reverse scan; computational cuts require both successor inputs.

    A true terminal masks both inputs. The caller owns event boundaries and
    reward attribution. Observation text is never silently converted to Steps.
    """
    if not math.isfinite(next_value) or not math.isfinite(next_advantage):
        raise ValueError('Continuation inputs must be finite')
    out=[]; nv=next_value; na=next_advantage
    for s in reversed(steps):
        g=0.0 if s.terminal else s.gamma
        a=s.reward+g*nv-s.value+g*s.lam*na
        out.append(a);nv=s.value;na=a
    return out[::-1]

def shape(rewards: Sequence[float], potentials: Sequence[float],
          discounts: Sequence[float]) -> list[float]:
    """Potential shaping for a complete episode with zero terminal potential."""
    if len(potentials)!=len(rewards)+1 or len(discounts)!=len(rewards):
        raise ValueError('Invalid complete-episode dimensions')
    if potentials[-1]!=0:
        raise ValueError('Complete episodes require zero terminal potential')
    if not all(math.isfinite(x) for x in (*rewards,*potentials,*discounts)):
        raise ValueError('Nonfinite input')
    if not all(0<=g<=1 for g in discounts):
        raise ValueError('Invalid discount')
    return [r+g*potentials[i+1]-potentials[i]
            for i,(r,g) in enumerate(zip(rewards,discounts))]

def discounted_return(rewards: Sequence[float], discounts: Sequence[float]) -> float:
    if len(rewards)!=len(discounts):raise ValueError('Length mismatch')
    total=0.0;weight=1.0
    for r,g in zip(rewards,discounts):total+=weight*r;weight*=g
    return total

def spr(success: bool, passed: int, divisor: float) -> float:
    if type(success) is not bool:raise TypeError('Success must be a protected boolean')
    if type(passed) is not int or passed<0:raise ValueError('Invalid passed count')
    if not math.isfinite(divisor) or divisor<=0:raise ValueError('Invalid divisor')
    return float(success)+passed/(passed+divisor)
