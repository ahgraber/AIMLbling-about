"""Identify optimal Damerau-Levenshtein edits to transition from s1 to s2.

Ref: https://github.com/nickeldan/dam_lev
"""

# %%
import dataclasses
from enum import StrEnum, auto
import functools
from typing import List, Sequence


# %%
class Operation(StrEnum):
    """Valid Damerau-Levenshtein edit operations."""

    delete = auto()
    insert = auto()
    substitute = auto()
    transpose = auto()


@dataclasses.dataclass(repr=False)
class EditOp:
    """Damerau-Levenshtein edit to transform from s1 to s2."""

    op: Operation
    s1_idx: int
    s2_idx: int = -1

    def __repr__(self) -> str:
        """Reader-friendly representation."""
        if self.s2_idx < 0:
            if self.op == Operation.delete:
                edit = f"s1[{self.s1_idx}]"
            elif self.op == Operation.transpose:
                edit = f"s1[{self.s1_idx}] with s1[{self.s1_idx+1}]"
            else:
                raise ValueError(f"Unexpected operation: {self.__dict__}")
        else:
            if self.op == Operation.insert:
                edit = f"s2[{self.s2_idx}] at s1[{self.s1_idx}]"
            elif self.op == Operation.substitute:
                edit = f"s1[{self.s1_idx}] with s2[{self.s2_idx}]"
            else:
                raise ValueError(f"Unexpected operation: {self.__dict__}")
        return f"{self.op.__str__()} {edit}"

    def to_dict(self):
        """Convert object to dictionary."""
        d = self.__dict__.copy()
        d["op"] = self.op.value
        return d

    todict = to_dict
    asdict = to_dict


# %%
class DamerauLevenshtein:
    """Identify the distance and edits to transform s1 into s2 using theDamerau-Levenshtein algorithm."""

    class Trace:
        """Trace of optimal edit operations to transform s1 into s2."""

        def __init__(self) -> None:
            self.edits: List[EditOp] = []
            self.score = -1

        def update(self, edits: List[EditOp]) -> None:
            """Update the optimal edit trace."""
            length = len(edits)
            if length < self.score or self.score < 0:
                self.edits = edits
                self.score = length

    def __init__(self, s1: Sequence, s2: Sequence):
        self.s1: Sequence = s1
        self.s2: Sequence = s2

        self.edits, self.distance = self._calculate()

    @functools.lru_cache(maxsize=None)  # NOQA: B019
    def _chain(self, i: int, j: int):
        if i < 0 and j < 0:
            return []

        trace = self.Trace()

        if i >= 0 and j >= 0:
            char1 = self.s1[i]
            char2 = self.s2[j]
            different_chars = char1 != char2
        else:
            different_chars = True

        if (
            i >= 1  # fmt: skip
            and j >= 1
            and char1 == self.s2[j - 1]
            and self.s1[i - 1] == char2
        ):
            trace.update(self._chain(i - 2, j - 2) + [EditOp(op=Operation.transpose, s1_idx=i - 1)])

        if different_chars:
            if i >= 0:
                trace.update(self._chain(i - 1, j) + [EditOp(op=Operation.delete, s1_idx=i)])

            if j >= 0:
                trace.update(self._chain(i, j - 1) + [EditOp(op=Operation.insert, s1_idx=i + 1, s2_idx=j)])

        if i >= 0 and j >= 0:
            prev_value = self._chain(i - 1, j - 1)
            if different_chars:
                trace.update(prev_value + [EditOp(op=Operation.substitute, s1_idx=i, s2_idx=j)])
            else:
                trace.update(list(prev_value))

        return trace.edits

    def _calculate(self):
        """Calculate the edit operations and edit distance from s1 to s2."""
        _trace = self._chain(len(self.s1) - 1, len(self.s2) - 1)
        return _trace, len(_trace)


# %%
