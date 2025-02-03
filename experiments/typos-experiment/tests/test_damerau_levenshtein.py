# %%
import pytest

from src.damerau_levenshtein import *


# %%
def test_diff_empty() -> None:
    dl = DamerauLevenshtein("", "")
    assert isinstance(dl.edits, list)
    assert not dl.edits


class TestOps:
    def test_transposition(self) -> None:
        dl = DamerauLevenshtein("ab", "ba")
        assert dl.edits == [EditOp(op=Operation.transpose, s1_idx=0)]

    def test_substitution(self) -> None:
        dl = DamerauLevenshtein("a", "b")
        assert dl.edits == [EditOp(op=Operation.substitute, s1_idx=0, s2_idx=0)]

    def test_insertion(self) -> None:
        dl = DamerauLevenshtein("a", "ab")
        assert dl.edits == [EditOp(op=Operation.insert, s1_idx=1, s2_idx=1)]

    def test_deletion(self) -> None:
        dl = DamerauLevenshtein("a", "")
        assert dl.edits == [EditOp(op=Operation.delete, s1_idx=0)]


class TestOpCombos:
    def test_transposition_then_substitution(self) -> None:
        dl = DamerauLevenshtein("abc", "bad")
        assert dl.edits == [
            EditOp(op=Operation.transpose, s1_idx=0),
            EditOp(op=Operation.substitute, s1_idx=2, s2_idx=2),
        ]

    def test_transposition_then_insertion(self) -> None:
        dl = DamerauLevenshtein("ab", "bac")
        assert dl.edits == [
            EditOp(op=Operation.transpose, s1_idx=0),
            EditOp(op=Operation.insert, s1_idx=2, s2_idx=2),
        ]

    def test_transposition_then_deletion(self) -> None:
        dl = DamerauLevenshtein("abc", "ba")
        assert dl.edits == [
            EditOp(op=Operation.transpose, s1_idx=0),
            EditOp(op=Operation.delete, s1_idx=2),
        ]

    def test_substitution_then_transposition(self) -> None:
        dl = DamerauLevenshtein("abc", "xcb")
        assert dl.edits == [
            EditOp(op=Operation.substitute, s1_idx=0, s2_idx=0),
            EditOp(op=Operation.transpose, s1_idx=1),
        ]

    def test_substitution_then_insertion(self) -> None:
        dl = DamerauLevenshtein("a", "bc")
        assert dl.edits == [
            EditOp(op=Operation.substitute, s1_idx=0, s2_idx=0),
            EditOp(op=Operation.insert, s1_idx=1, s2_idx=1),
        ]

    def test_substitution_then_deletion(self) -> None:
        dl = DamerauLevenshtein("ab", "c")
        assert dl.edits == [
            EditOp(op=Operation.substitute, s1_idx=0, s2_idx=0),
            EditOp(op=Operation.delete, s1_idx=1),
        ]

    def test_insertion_then_transposition(self) -> None:
        dl = DamerauLevenshtein("ab", "cba")
        assert dl.edits == [
            EditOp(op=Operation.insert, s1_idx=0, s2_idx=0),
            EditOp(op=Operation.transpose, s1_idx=0),
        ]

    def test_insertion_then_substitution(self) -> None:
        dl = DamerauLevenshtein("abc", "xabd")
        assert dl.edits == [
            EditOp(op=Operation.insert, s1_idx=0, s2_idx=0),
            EditOp(op=Operation.substitute, s1_idx=2, s2_idx=3),
        ]

    def test_insertion_then_deletion(self) -> None:
        dl = DamerauLevenshtein("abc", "xab")
        assert dl.edits == [
            EditOp(op=Operation.insert, s1_idx=0, s2_idx=0),
            EditOp(op=Operation.delete, s1_idx=2),
        ]

    def test_deletion_then_transposition(self) -> None:
        dl = DamerauLevenshtein("abc", "cb")
        assert dl.edits == [
            EditOp(op=Operation.delete, s1_idx=0),
            EditOp(op=Operation.transpose, s1_idx=1),
        ]

    def test_deletion_then_substitution(self) -> None:
        dl = DamerauLevenshtein("abcd", "bce")
        assert dl.edits == [
            EditOp(op=Operation.delete, s1_idx=0),
            EditOp(op=Operation.substitute, s1_idx=3, s2_idx=2),
        ]

    def test_deletion_then_insertion(self) -> None:
        dl = DamerauLevenshtein("abc", "bcd")
        assert dl.edits == [
            EditOp(op=Operation.delete, s1_idx=0),
            EditOp(op=Operation.insert, s1_idx=3, s2_idx=2),
        ]

    def test_different_types(self) -> None:
        dl = DamerauLevenshtein("abc", ["a", "b"])
        assert dl.edits == [EditOp(op=Operation.delete, s1_idx=2)]

    def test_nonhashable_elements(self) -> None:
        dl = DamerauLevenshtein([[1, 2], [3, 4]], [[1], [3, 4]])
        assert dl.edits == [EditOp(op=Operation.substitute, s1_idx=0, s2_idx=0)]


# %%
class TestDistance:
    @pytest.fixture
    def testcases(self):
        return [
            ("test", "text", 1),
            ("test", "tset", 1),
            ("test", "qwy", 4),
            ("test", "testit", 2),
            ("test", "tesst", 1),
            ("test", "tet", 1),
            ("cat", "hat", 1),
            ("Niall", "Neil", 3),
            ("aluminum", "Catalan", 7),
            ("ATCG", "TAGC", 2),
            ("ab", "ba", 1),
            ("ab", "cde", 3),
            ("ab", "ac", 1),
            ("ab", "bc", 2),
            ("gifts", "profit", 5),
            ("spartan", "part", 3),
            ("republican", "democrat", 8),
            ("fish", "ifsh", 1),
            ("staes", "states", 1),
            ("plasma", "plasma", 0),
            ("", "", 0),
            ("the", "", 3),
            ("", "the", 3),
        ]

    def test_distance(self, testcases):
        for case in testcases:
            dl = DamerauLevenshtein(case[0], case[1])
            assert dl.distance == case[2], f"Failed {case} - expected {case[2]}, got {dl.distance}"


# %%
