from __future__ import annotations

from lsa.canonical import model_hash, run_hash
from tests.test_lsa_canonical import _instance


def test_identical_inputs_generate_identical_model_and_run_hashes() -> None:
    instance = _instance("block", "B")
    model = {"family": "looped_decoder_iso_shape_v0", "hidden_size": 416}
    run = {"seed": 101, "optimizer": "adamw", "exposures": 1000}

    assert model_hash(instance, model) == model_hash(instance, dict(model))
    assert run_hash(instance, model, run) == run_hash(instance, dict(model), dict(run))


def test_seed_changes_run_hash_but_not_model_hash() -> None:
    instance = _instance("block", "B")
    model = {"family": "looped_decoder_iso_shape_v0", "hidden_size": 416}

    assert model_hash(instance, model) == model_hash(instance, model)
    assert run_hash(instance, model, {"seed": 1}) != run_hash(
        instance, model, {"seed": 2}
    )
