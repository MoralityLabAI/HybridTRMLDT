"""CCA-10 analytical bounds: finite state dominance checks."""
from math import comb, isclose
import pytest
from research_gym.core.cca_optimistic_bound import (
    action_count_vectors, optimistic_moral_upper,
)

def test_count_vectors_cover_all_lengths():
    for h in range(8):
        vectors=action_count_vectors(h)
        assert len(vectors)==comb(h+5,5)
        assert len(set(vectors))==len(vectors)
        assert all(sum(v)<=h for v in vectors)

def test_zero_horizon_matches_terminal_value():
    for t in range(-3,4):
        for e in range(5):
            for h in range(5):
                for scene in range(6):
                    expected=min(1.,max(0.,(3*t+2*e-3*h+scene+21.)/43.))
                    assert isclose(optimistic_moral_upper(t,e,h,scene,0),expected)

def test_optimistic_upper_is_nondecreasing_in_horizon():
    for t,e,h,scene in [(-3,0,4,0),(0,2,3,2),(3,4,0,4),(-1,1,2,3)]:
        values=[optimistic_moral_upper(t,e,h,scene,k) for k in range(7)]
        assert values==sorted(values)
        assert all(0<=v<=1 for v in values)

def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        action_count_vectors(-1)
    with pytest.raises(ValueError):
        optimistic_moral_upper(10,1,1,1,3)

def test_bound_dominates_exact_dynamic_programming_on_all_states_and_horizons():
    from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv
    from experiments.cca05_certified_trm_repair.src import semantics as M
    env=CoupledStoryworldEnv()
    checked=0
    for s in env.all_states():
        for k in range(6):
            upper=optimistic_moral_upper(s.trust,s.evidence,s.heat,s.scene,k)
            best=M.best_value(s,k,"moral_optimization")
            assert upper+1e-12>=best,(s,k,best,upper)
            checked+=1
    assert checked==6300
