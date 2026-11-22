"""Sets of tests are Python ints used as bitsets: bit ``i`` stands for test number ``i``.

Bitsets keep the coverage atlas compact (one int per covered line) and make unions and
intersections over thousands of tests a single machine-level operation.
"""

from __future__ import annotations

from typing import Iterable, Iterator


def members(bitset: int) -> Iterator[int]:
    """Indices of the set bits, lowest first."""
    while bitset:
        lowest = bitset & -bitset
        yield lowest.bit_length() - 1
        bitset ^= lowest


def count(bitset: int) -> int:
    return bitset.bit_count()


def from_indices(indices: Iterable[int]) -> int:
    bitset = 0
    for index in indices:
        bitset |= 1 << index
    return bitset


def has(bitset: int, index: int) -> bool:
    return (bitset >> index) & 1 == 1


def full(size: int) -> int:
    return (1 << size) - 1
