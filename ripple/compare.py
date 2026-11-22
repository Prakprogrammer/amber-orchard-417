"""Patches that change exactly one comparison, judged against recorded comparison results.

The comparison atlas (built by ``forge.comparisons``) knows, for every single comparison
``a OP b`` in the pinned sqlglot and every test, what each of the ten comparison operators
would have returned on every evaluation the test performed; where one operand is a literal it
also keeps the values the other operand took. A patch that only swaps the operator, or only
changes that literal, can alter a test only by flipping one of those recorded evaluations.
Until then the test executes exactly as it does on the pinned commit.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass

from .syntax import overlaps_edit

# Operator codes: the order is shared with the atlas, so it must not change.
OPERATORS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)
IDENTITY_CODES = {6, 7}
_EVALUATE = (
    operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge,
    operator.is_, operator.is_not, lambda a, b: a in b, lambda a, b: a not in b,
)
# Two bits per operator in a recorded result word: 0 False, 1 True, 2 raised, 3 unknown.
RAISED, UNKNOWN = 2, 3


def evaluate(code: int, left, right):
    return _EVALUATE[code](left, right)


def operator_code(node: ast.cmpop) -> int:
    for code, kind in enumerate(OPERATORS):
        if isinstance(node, kind):
            return code
    return -1


def outcome_of(word: int, code: int) -> int:
    return (word >> (2 * code)) & 3


def is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, (bool, int, float, str, bytes))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    if isinstance(node, ast.Tuple):
        return all(is_literal(item) for item in node.elts)
    return False


@dataclass
class ComparisonEdit:
    where: tuple[int, int, int, int]  # (line, column, end line, end column) in the old file
    old_code: int
    new_code: int
    literal_side: str = ""  # "L" or "R" when that literal operand's value changed
    old_literal: object = None
    new_literal: object = None

    def describe(self) -> str:
        text = f"{OPERATORS[self.old_code].__name__}->{OPERATORS[self.new_code].__name__}"
        if self.literal_side:
            text += f" {self.old_literal!r}->{self.new_literal!r}"
        return text


class _NotOneComparison(Exception):
    pass


def _same_node(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def find_single_comparison_edit(old_tree: ast.AST, new_tree: ast.AST, removed: set[int] | None = None,
                                added: set[int] | None = None) -> ComparisonEdit | None:
    """The one comparison whose operator and/or literal operand is all that differs between
    the trees; None for every other kind of edit. Statements away from the edited lines
    (``removed`` in old numbering, ``added`` in new numbering) are skipped."""
    edits: list[ComparisonEdit] = []

    def walk(old, new) -> None:
        if type(old) is not type(new):
            raise _NotOneComparison
        if isinstance(old, list):
            if len(old) != len(new):
                raise _NotOneComparison
            for old_item, new_item in zip(old, new):
                if (isinstance(old_item, ast.stmt) and isinstance(new_item, ast.stmt)
                        and not overlaps_edit(old_item, new_item, removed, added)):
                    continue
                walk(old_item, new_item)
            return
        if not isinstance(old, ast.AST):
            if old != new:
                raise _NotOneComparison
            return
        if isinstance(old, ast.Compare) and len(old.ops) == 1 and len(new.ops) == 1 and _record(old, new):
            return
        for field_name in old._fields:
            walk(getattr(old, field_name, None), getattr(new, field_name, None))

    def _record(old: ast.Compare, new: ast.Compare) -> bool:
        """Note this comparison if it is edited (then walk only its unchanged operand)."""
        old_code, new_code = operator_code(old.ops[0]), operator_code(new.ops[0])
        left_same = _same_node(old.left, new.left)
        right_same = _same_node(old.comparators[0], new.comparators[0])
        side, literals = "", None
        if not left_same and right_same and is_literal(old.left) and is_literal(new.left):
            side, literals = "L", (old.left, new.left)
        elif not right_same and left_same and is_literal(old.comparators[0]) and is_literal(new.comparators[0]):
            side, literals = "R", (old.comparators[0], new.comparators[0])
        if old_code == new_code and not side:
            return False
        if old_code < 0 or new_code < 0:
            raise _NotOneComparison
        edit = ComparisonEdit((old.lineno, old.col_offset, old.end_lineno, old.end_col_offset), old_code, new_code)
        if side:
            try:
                edit.old_literal, edit.new_literal = ast.literal_eval(literals[0]), ast.literal_eval(literals[1])
            except (ValueError, TypeError, SyntaxError):
                raise _NotOneComparison
            edit.literal_side = side
        edits.append(edit)
        if len(edits) > 1:
            raise _NotOneComparison
        if side == "L":
            walk(old.comparators, new.comparators)
        elif side == "R":
            walk(old.left, new.left)
        else:
            walk(old.left, new.left)
            walk(old.comparators, new.comparators)
        return True

    try:
        walk(old_tree, new_tree)
    except (_NotOneComparison, RecursionError):
        return None
    return edits[0] if len(edits) == 1 else None


class ComparisonAtlas:
    """Recorded comparison results of the pinned tree, per site and test."""

    def __init__(self, payload: dict, test_ids: list[str]):
        if payload["test_ids"] != test_ids:
            raise ValueError("the comparison atlas was recorded for a different test list")
        self.sites: dict[tuple, tuple[int, str]] = payload["sites"]  # site -> (operator code, literal side)
        self.import_time: set = payload["import_time"]
        self.results: dict[tuple, dict[int, int]] = payload["results"]  # site -> {result word: tests}
        self.operands: dict[tuple, dict[object, int]] = payload["operands"]  # site -> {other operand: tests}
        self.operands_lost: dict[tuple, int] = payload["operands_lost"]  # site -> tests whose values were not kept
        self.untrusted: int = payload["untrusted"]
        self.files: set[str] = set(payload["files"])

    def tests_flipped_by(self, path: str, edit: ComparisonEdit) -> int | None:
        """Tests for which some recorded evaluation comes out differently after the edit, or
        None when the atlas cannot tell (unknown site, evaluated during import...). Untrusted
        tests are left to the caller."""
        if path not in self.files:
            return None
        site = (path,) + edit.where
        recorded = self.sites.get(site)
        if recorded is None or site in self.import_time:
            return None
        code, literal_side = recorded
        if code != edit.old_code:
            return None
        words = self.results.get(site, {})
        if not edit.literal_side:
            return self._operator_flips(words, edit)
        if edit.literal_side != literal_side or {edit.old_code, edit.new_code} & IDENTITY_CODES:
            return None  # identity tests on literals: equal recorded values were merged
        return self._literal_flips(site, words, edit)

    @staticmethod
    def _operator_flips(words: dict[int, int], edit: ComparisonEdit) -> int:
        flipped = 0
        for word, tests in words.items():
            before, after = outcome_of(word, edit.old_code), outcome_of(word, edit.new_code)
            if before >= RAISED or after >= RAISED or before != after:
                flipped |= tests
        return flipped

    def _literal_flips(self, site: tuple, words: dict[int, int], edit: ComparisonEdit) -> int:
        evaluated = 0
        for tests in words.values():
            evaluated |= tests
        flipped = kept = 0
        for value, tests in self.operands.get(site, {}).items():
            kept |= tests
            try:
                if edit.literal_side == "R":
                    before = evaluate(edit.old_code, value, edit.old_literal)
                    after = evaluate(edit.new_code, value, edit.new_literal)
                else:
                    before = evaluate(edit.old_code, edit.old_literal, value)
                    after = evaluate(edit.new_code, edit.new_literal, value)
                unchanged = type(before) is bool and type(after) is bool and before == after
            except Exception:
                unchanged = False
            if not unchanged:
                flipped |= tests
        return flipped | self.operands_lost.get(site, 0) | (evaluated & ~kept)
