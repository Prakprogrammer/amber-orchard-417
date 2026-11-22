"""Static questions about one Python source file, answered from its AST.

``Outline`` answers "what encloses line N?" lazily: it walks only the chain of nodes whose
span contains the line, so a large module costs one parse and a handful of short walks.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Union

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_CALLABLE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_EXPRESSION_MODULE_ALIASES = ("exp", "expressions")


@dataclass
class FunctionScope:
    qualname: str
    first_line: int  # first decorator line, else the ``def`` line
    def_line: int
    body_line: int
    last_line: int
    name: str
    node: ast.AST
    owner_class: str | None = None  # qualname of the class this is directly a method of

    @property
    def is_lambda(self) -> bool:
        return self.name == "<lambda>"


@dataclass
class ClassScope:
    qualname: str
    first_line: int
    last_line: int
    name: str


Scope = Union[FunctionScope, ClassScope, None]  # None: module level


def _first_line(node: ast.AST) -> int:
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", None) or ()])


def _spans(node: ast.AST, line: int) -> bool | None:
    """Whether the node's lines include ``line``; None when the node has no span of its own."""
    if not hasattr(node, "lineno") or getattr(node, "end_lineno", None) is None:
        return None
    return _first_line(node) <= line <= node.end_lineno


def _qualname(chain: list[ast.AST]) -> tuple[str, str | None]:
    """(dotted name of the innermost def/class/lambda in ``chain``, its class if a method)."""
    parts: list[tuple[str, bool]] = []  # (name, is_callable)
    for node in chain:
        if isinstance(node, ast.ClassDef):
            parts.append((node.name, False))
        elif isinstance(node, _DEF_NODES):
            parts.append((node.name, True))
        elif isinstance(node, ast.Lambda):
            parts.append(("<lambda>", True))
    qualname = ""
    for position, (name, _) in enumerate(parts):
        if position == 0:
            qualname = name
        else:
            qualname += (".<locals>." if parts[position - 1][1] else ".") + name
    method_of = None
    if len(parts) >= 2 and not parts[-2][1] and parts[-1][1]:
        method_of = qualname.rsplit(".", 1)[0]
    return qualname, method_of


def expression_classes_named(node: ast.AST) -> set[str]:
    """Names X in ``exp.X`` / ``expressions.X`` references (capitalised: classes)."""
    return {
        sub.attr for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
        and sub.value.id in _EXPRESSION_MODULE_ALIASES and sub.attr[:1].isupper()
    }


class Outline:
    def __init__(self, tree: ast.Module | None):
        self.tree = tree
        self._definitions: dict[str, Scope] | None = None

    @property
    def broken(self) -> bool:
        return self.tree is None

    # ---- the chain of nodes around one line -------------------------------------------
    def chain_to(self, line: int) -> list[ast.AST]:
        chain: list[ast.AST] = []
        node = self.tree
        while True:
            frontier = list(ast.iter_child_nodes(node))
            inner = None
            while frontier:
                child = frontier.pop(0)
                spans = _spans(child, line)
                if spans is None:
                    frontier[:0] = list(ast.iter_child_nodes(child))
                elif spans:
                    inner = child
                    break
            if inner is None:
                return chain
            chain.append(inner)
            node = inner

    def scope_at(self, line: int) -> Scope:
        chain = self.chain_to(line)
        for depth in range(len(chain) - 1, -1, -1):
            node = chain[depth]
            if isinstance(node, ast.Lambda):
                qualname, _ = _qualname(chain[:depth + 1])
                return FunctionScope(qualname, node.lineno, node.lineno, node.lineno, node.end_lineno,
                                     "<lambda>", node, self._lambda_owner(chain, depth))
            if isinstance(node, _DEF_NODES):
                qualname, method_of = _qualname(chain[:depth + 1])
                return FunctionScope(qualname, _first_line(node), node.lineno, node.body[0].lineno,
                                     node.end_lineno, node.name, node, method_of)
            if isinstance(node, ast.ClassDef):
                qualname, _ = _qualname(chain[:depth + 1])
                return ClassScope(qualname, _first_line(node), node.end_lineno, node.name)
        return None

    @staticmethod
    def _lambda_owner(chain: list[ast.AST], depth: int) -> str | None:
        """A lambda sitting directly in a class body (say, in a table) belongs to that class."""
        for outer in range(depth - 1, -1, -1):
            if isinstance(chain[outer], _CALLABLE_NODES):
                return None
            if isinstance(chain[outer], ast.ClassDef):
                return _qualname(chain[:outer + 1])[0]
        return None

    def statement_lines(self, line: int) -> tuple[int, int] | None:
        """The lines whose execution shows that this line's statement ran: all of a simple
        statement, or the header of a compound one. None for a line in a def/class body that
        belongs to no statement (blank or comment)."""
        chain = self.chain_to(line)
        statement = next((node for node in reversed(chain) if isinstance(node, ast.stmt)), None)
        if statement is None:
            return None
        body = getattr(statement, "body", None)
        if not (isinstance(body, list) and body and isinstance(body[0], ast.AST)):
            return statement.lineno, statement.end_lineno
        header_end = body[0].lineno - 1
        if isinstance(statement, (*_DEF_NODES, ast.ClassDef)):
            header_end = max(header_end, statement.lineno)
            if line > header_end:
                return None
        return statement.lineno, max(header_end, statement.lineno)

    def table_entry_classes(self, line: int) -> frozenset[str] | None:
        """Expression classes named by the dict entry (key and value) holding this line; None
        when the line is in no entry, in a ``**spread``, or the entry names none."""
        for node in reversed(self.chain_to(line)):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    parts = [part for part in (key, value) if part is not None]
                    if min(map(_first_line, parts)) <= line <= max(p.end_lineno for p in parts):
                        if key is None:
                            return None
                        named = set().union(*(expression_classes_named(part) for part in parts))
                        return frozenset(named) or None
                return None
            if isinstance(node, (*_CALLABLE_NODES, ast.ClassDef)):
                return None
        return None

    def assigned_name(self, line: int) -> str | None:
        """NAME when the line is part of a plain ``NAME = ...`` (or annotated) statement at
        module level or directly in a class body, with no function inside its value."""
        chain = self.chain_to(line)
        in_namespace = True  # the module body is a namespace
        for depth, node in enumerate(chain):
            if isinstance(node, ast.stmt) and in_namespace:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    nested_def = any(isinstance(x, (ast.Lambda, ast.FunctionDef)) for x in chain[depth + 1:])
                    if len(targets) == 1 and isinstance(targets[0], ast.Name) and not nested_def:
                        return targets[0].id
                if not isinstance(node, ast.ClassDef):
                    return None
            in_namespace = isinstance(node, ast.ClassDef)
        return None

    # ---- definitions by qualified name ------------------------------------------------
    def _collect_definitions(self) -> dict[str, Scope]:
        found: dict[str, Scope] = {}

        def visit(statements, prefix: str, enclosing_class: str | None) -> None:
            for node in statements:
                if isinstance(node, _DEF_NODES):
                    qualname = prefix + node.name
                    found.setdefault(qualname, FunctionScope(
                        qualname, _first_line(node), node.lineno, node.body[0].lineno, node.end_lineno,
                        node.name, node, enclosing_class))
                    visit(node.body, qualname + ".<locals>.", None)
                elif isinstance(node, ast.ClassDef):
                    qualname = prefix + node.name
                    found.setdefault(qualname, ClassScope(qualname, _first_line(node), node.end_lineno, node.name))
                    visit(node.body, qualname + ".", qualname)
                else:  # definitions under if/for/while/with/try stay in the same namespace
                    for block in ("body", "orelse", "finalbody", "handlers"):
                        inner = getattr(node, block, None)
                        if isinstance(inner, list):
                            visit([x for x in inner if isinstance(x, ast.AST)], prefix, enclosing_class)

        visit(self.tree.body if self.tree is not None else [], "", None)
        return found

    def _definition(self, qualname: str):
        if self._definitions is None:
            self._definitions = self._collect_definitions()
        return self._definitions.get(qualname)

    def function(self, qualname: str) -> FunctionScope | None:
        found = self._definition(qualname)
        return found if isinstance(found, FunctionScope) else None

    def klass(self, qualname: str) -> ClassScope | None:
        found = self._definition(qualname)
        return found if isinstance(found, ClassScope) else None


def outline(source: str | bytes) -> Outline:
    try:
        return Outline(ast.parse(source))
    except (SyntaxError, ValueError):
        return Outline(None)


_COMMENT_OR_SPACE = re.compile(r"\s+|#.*")


def could_be_cosmetic(removed: list[str], added: list[str]) -> bool:
    """Cheap first test: do the removed and added rows differ only in layout and comments?"""
    strip = lambda rows: "".join(_COMMENT_OR_SPACE.sub("", row) for row in rows)  # noqa: E731
    return strip(removed) == strip(added)


def same_tree(old: str | bytes, new: str | bytes) -> bool:
    try:
        return ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))
    except (SyntaxError, ValueError):
        return False


def overlaps_edit(old_stmt: ast.stmt, new_stmt: ast.stmt, removed: set[int] | None, added: set[int]) -> bool:
    """Do these paired statements touch an edited line? Statements away from every edit parse
    the same on both sides and need no comparison."""
    if removed is None:
        return True
    first, last = old_stmt.lineno, old_stmt.end_lineno or old_stmt.lineno
    if any(first <= line <= last for line in removed):
        return True
    first, last = new_stmt.lineno, new_stmt.end_lineno or new_stmt.lineno
    return any(first <= line <= last for line in added)


class _EditOutsideFunctions(Exception):
    pass


def edited_functions(old_tree, new_tree, removed: set[int], added: set[int]) -> list[ast.AST] | None:
    """The innermost functions/lambdas (old-tree nodes) whose bodies hold every difference
    between the two trees. None when anything differs outside a body (module or class level,
    a signature, a decorator, a default) or a docstring changes (``__doc__`` is set once)."""
    holders: list[ast.AST] = []

    def differ(holder) -> None:
        if holder is None:
            raise _EditOutsideFunctions
        holders.append(holder)

    def compare(old, new, holder) -> None:
        if type(old) is not type(new):
            differ(holder)
        elif isinstance(old, list):
            if len(old) != len(new):
                differ(holder)
                return
            for old_item, new_item in zip(old, new):
                if (isinstance(old_item, ast.stmt) and isinstance(new_item, ast.stmt)
                        and not overlaps_edit(old_item, new_item, removed, added)):
                    continue
                compare(old_item, new_item, holder)
        elif not isinstance(old, ast.AST):
            if old != new:
                differ(holder)
        else:
            for field_name in old._fields:
                inner = holder
                if field_name == "body" and isinstance(old, _DEF_NODES):
                    _check_docstring(old, new)
                    inner = old
                elif field_name == "body" and isinstance(old, ast.Lambda):
                    inner = old
                compare(getattr(old, field_name, None), getattr(new, field_name, None), inner)

    try:
        compare(old_tree, new_tree, None)
    except (_EditOutsideFunctions, RecursionError):
        return None
    unique: list[ast.AST] = []
    for node in holders:
        if not any(node is seen for seen in unique):
            unique.append(node)
    return unique


def _check_docstring(old: ast.AST, new: ast.AST) -> None:
    body = old.body
    opens_with_string = (body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                         and isinstance(body[0].value.value, str))
    if opens_with_string and ast.get_docstring(old, clean=False) != ast.get_docstring(new, clean=False):
        raise _EditOutsideFunctions


def runtime_imports(source: str | bytes, module: str, is_package: bool) -> list[tuple[str, bool, int, int]]:
    """(imported module name, imported inside a function?, statement first line, last line) for
    every import that can run; imports guarded by ``if TYPE_CHECKING:`` never run."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    package = module if is_package else module.rpartition(".")[0]
    found: list[tuple[str, bool, int, int]] = []

    def targets(node) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        if node.level:
            base = package.split(".")
            if node.level > 1:
                base = base[: len(base) - (node.level - 1)]
            origin = ".".join(base + ([node.module] if node.module else []))
        else:
            origin = node.module or ""
        return [origin] + [f"{origin}.{alias.name}" for alias in node.names]

    def type_checking_guard(test) -> bool:
        return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))

    def statement(node, in_function: bool) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.extend((name, in_function, node.lineno, node.end_lineno) for name in targets(node))
        elif isinstance(node, ast.If) and type_checking_guard(node.test):
            for other in node.orelse:
                statement(other, in_function)
        else:
            children(node, in_function or isinstance(node, _CALLABLE_NODES))

    def children(node, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            statement(child, in_function)

    children(tree, False)
    return found
