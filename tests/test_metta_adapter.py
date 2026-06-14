import pytest

from research_gym.adapters.metta_adapter import FakeMettaAdapter, MettaAdapterProtocol, ParsedMettaProgram
from research_gym.core.metta_frames import ExecutionFrame


DIRECTIVES = """
!exec rule=arith_add_heat before=heat:1,delta:2 after=heat:3,delta:2 invariants=arithmetic,bounds
!repair diagnostic=unknown_symbol buggy="(bad)" repaired="(good)" passed=true
""".strip()


def accepts_protocol(adapter: MettaAdapterProtocol) -> MettaAdapterProtocol:
    return adapter


def test_fake_adapter_satisfies_protocol_shape():
    adapter = accepts_protocol(FakeMettaAdapter(source="unit"))
    program = adapter.parse(DIRECTIVES)

    assert isinstance(program, ParsedMettaProgram)
    assert adapter.typecheck(program)


def test_fake_adapter_extracts_synthetic_frames():
    adapter = FakeMettaAdapter(source="unit")
    program = adapter.parse(DIRECTIVES)
    frames = adapter.extract_frames(program)

    assert len(frames) == 2
    assert isinstance(frames[0], ExecutionFrame)
    assert frames[0].source == "unit"


def test_fake_adapter_execute_returns_state_wrapper():
    adapter = FakeMettaAdapter(source="unit")
    program = adapter.parse(DIRECTIVES)

    result = adapter.execute(program, {"heat": 1})

    assert result == {"state": {"heat": 1}, "executed": True, "source": "unit"}


def test_fake_adapter_typecheck_rejects_bad_directive():
    adapter = FakeMettaAdapter(source="unit")
    program = adapter.parse("!unknown rule=x")

    assert not adapter.typecheck(program)
    with pytest.raises(ValueError, match="failed fake typecheck"):
        adapter.execute(program, {})
