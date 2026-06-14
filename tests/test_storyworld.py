from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState
from research_gym.core.typed_soundness import SoundnessType


def test_target_predicate_is_reachable_from_good_state():
    env = CoupledStoryworldEnv()
    state = StoryState(trust=2, evidence=3, heat=0, scene=4)
    reachable, path = env.reachable(state, horizon=2, other_mode="frozen")
    assert reachable
    assert path


def test_label_frame_returns_known_soundness_type():
    env = CoupledStoryworldEnv()
    frame = env.label_frame(StoryState(trust=-3, evidence=0, heat=4, scene=4), horizon=1, frame_id="x")
    assert frame.label in set(SoundnessType)


def test_model_sound_case_exists_somewhere():
    env = CoupledStoryworldEnv()
    found = False
    for state in env.all_states():
        frame = env.label_frame(state, horizon=6, frame_id="scan")
        if frame.label == SoundnessType.MODEL_SOUND_DEAD:
            found = True
            break
    assert found
