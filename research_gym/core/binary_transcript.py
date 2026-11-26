"""Fixed-schedule binary transcript reference for LOGRANK-BPD-01."""
from dataclasses import dataclass
from typing import Callable, Literal

Party = Literal['A', 'B']
BitPolicy = Callable[[object, tuple[int, ...]], int]


def _bit(value: int) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError('expected an integer bit')
    return value


@dataclass(frozen=True)
class Receipt:
    schedule: tuple[Party, ...]
    bits: tuple[int, ...]
    output: int

    @property
    def cost(self) -> int:
        return len(self.bits)


def execute(schedule: tuple[Party, ...], alice_input: object, bob_input: object,
            alice: BitPolicy, bob: BitPolicy, decode: BitPolicy,
            *, bit_cap: int) -> Receipt:
    """Charge every scheduled symbol; Bob's final local output is free.

    Callbacks must be isolated by the caller for an adversarial leakage audit.
    A Python callback can otherwise close over private state or shared memory.
    """
    if type(bit_cap) is not int or bit_cap < 0 or len(schedule) > bit_cap:
        raise ValueError('invalid bit cap')
    if any(s not in ('A', 'B') for s in schedule):
        raise ValueError('unknown speaker')
    bits: tuple[int, ...] = ()
    for speaker in schedule:
        message = alice(alice_input, bits) if speaker == 'A' else bob(bob_input, bits)
        bits += (_bit(message),)
    return Receipt(tuple(schedule), bits, _bit(decode(bob_input, bits)))


def replay(receipt: Receipt, alice_input: object, bob_input: object,
           alice: BitPolicy, bob: BitPolicy, decode: BitPolicy,
           *, bit_cap: int) -> bool:
    return execute(receipt.schedule, alice_input, bob_input, alice, bob, decode,
                   bit_cap=bit_cap) == receipt


def _self_test() -> None:
    def a(x, bits):
        return (x >> (2 * bits[0] + bits[1])) & 1

    def b(y, bits):
        return (y >> (1 - len(bits))) & 1

    def decode(y, bits):
        return bits[-1]

    for x in range(16):
        for y in range(4):
            r = execute(('B', 'B', 'A'), x, y, a, b, decode, bit_cap=3)
            assert r.cost == 3 and r.output == (x >> y) & 1
            assert replay(r, x, y, a, b, decode, bit_cap=3)
    for invalid in (2, -1, 1.0, True):
        try:
            execute(('A',), 0, 0, lambda x, bits: invalid, b, decode, bit_cap=1)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted non-bit')
    try:
        execute(('B', 'B', 'A'), 0, 0, a, b, decode, bit_cap=2)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted oversize schedule')
    print('64/64 index pairs replayed; 5 invalid cases rejected')


if __name__ == '__main__':
    _self_test()
