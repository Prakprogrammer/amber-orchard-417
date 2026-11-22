"""Shrinking what a candidate test has to run, decided before anything runs.

- ``edited_statements``: the pinned lines of every statement a patch touches.
- ``narrow_fixture_readers``: the readers of an edited .sql fixture run only the edited entries.
- ``DialectFamilies`` / ``narrow_to_dialects``: an edit confined to some dialects' modules runs
  only the fixture entries written for those dialects, plus the entries reaching the edit.
"""

from __future__ import annotations

import ast
import os
import re
import types
from dataclasses import dataclass, field

from .atlas import TestPasses
from .diffs import FileDiff, touched_old_lines
from .impact import Impact, ImpactAnalyzer
from .standins import FIXTURE_LOADERS, OnlyPasses, pass_kind

EXECUTOR_CLASS = "tests/test_executor.py::TestExecutor::"
TPC_QUERY_LISTS = {  # fixture -> the TestExecutor methods iterating it entry by entry
    "optimizer/tpc-h/tpc-h.sql": ("test_execute_tpch", "test_optimized_tpch"),
    "optimizer/tpc-ds/tpc-ds.sql": ("test_execute_tpcds",),
}
LOOPING_TESTS = {  # tests whose passes a dedicated stand-in selects (never by fixture entries)
    "tests/test_generator.py::TestGenerator::test_generator_memory",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_optimized_tpch",
}
_DIALECT_DIRS = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")
_SHARED_STEMS = {"__init__", "dialect", "base", "python"}
_REGISTRY_MODULES = {"sqlglot.dialects", "sqlglot.parsers", "sqlglot.generators", "sqlglot.typing"}
_CLASS_LINE_EDIT = re.compile(r"(?m)^[-+]\s*class\s")


@dataclass
class Reductions:
    """A commit's reduced tests: what each runs instead, and its planned CPU."""

    stand_ins: dict[str, object] = field(default_factory=dict)  # test id -> directive
    reduced_cpu: dict[int, float] = field(default_factory=dict)  # test -> unscaled CPU of its reduced run
    dialect_reduced: set[int] = field(default_factory=set)


# ---- edited statements -------------------------------------------------------------------
def edited_statements(analyzer: ImpactAnalyzer, diffs: list[FileDiff]) -> dict[str, set[int]] | None:
    """{path: pinned lines of every statement the patch touches}; None when unclear (a new
    file, a rename, or more lines added than removed: new statements have no pinned lines)."""
    edited: dict[str, set[int]] = {}
    try:
        for diff in diffs:
            path = diff.source
            if not path or path != diff.target:
                return None
            text = analyzer.source(path)
            if text is None:
                return None
            pinned = analyzer.pinned_outline(path, text)
            if diff.added_count > diff.removed_count:
                return None
            rewritten, borders = touched_old_lines(diff)
            lines = edited.setdefault(path, set())
            for line in rewritten | borders:
                first, last = pinned.statement_lines(line) or (line, line)
                lines.update(range(first, last + 1))
    except Exception:  # noqa
        return None
    return edited


# ---- .sql fixtures -------------------------------------------------------------------------
def fixture_entries(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """The entries of a .sql fixture as tests/helpers.py yields them: (lines, (sql, expected) pairs)."""
    kept = "\n".join(line for line in text.splitlines() if line and not line.startswith("--"))
    statements = kept.split(";")
    pairs = [(statements[i].strip(), statements[i + 1].strip()) for i in range(0, len(statements) - 1, 2)]
    return kept.splitlines(), pairs


def edited_fixture_entries(analyzer: ImpactAnalyzer, impact: Impact) -> dict[str, tuple[set[int], set[int]]]:
    """{fixture name: (edited line indices, edited pair indices)}. A fixture whose number of
    entries changes (or that appears or disappears) is left out."""
    edited = {}
    for path, text in impact.new_contents.items():
        if not (path.startswith("tests/fixtures/") and path.endswith(".sql")) or text is None:
            continue
        pinned = analyzer.source(path)
        if pinned is None:
            continue
        old_lines, old_pairs = fixture_entries(pinned)
        new_lines, new_pairs = fixture_entries(text)
        if len(old_lines) != len(new_lines) or len(old_pairs) != len(new_pairs):
            continue
        edited[path[len("tests/fixtures/"):]] = (
            {i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b},
            {i for i, (a, b) in enumerate(zip(old_pairs, new_pairs)) if a != b},
        )
    return edited


def narrow_fixture_readers(analyzer: ImpactAnalyzer, impact: Impact, tests: list[dict], data_readers: set[int],
                           candidates: set[int], passes: dict[str, TestPasses], run_cpu: list[float],
                           reductions: Reductions) -> set[int]:
    """Readers of edited .sql fixtures run only the edited entries, when they walk the file
    entry by entry (pass atlas, TPC query lists). TestExecutor tests that see the file only
    through their class setup are unaffected while it parses the same. Returns the data
    readers this settled (they are no longer listed unconditionally)."""
    edited = edited_fixture_entries(analyzer, impact)
    if not edited or any(not (p.startswith("tests/fixtures/") and p.endswith(".sql")) for p in impact.new_contents):
        return set()
    every_tpc_method = sum(TPC_QUERY_LISTS.values(), ())
    settled = set()
    for test in sorted(data_readers):
        test_id = tests[test]["id"]
        method = test_id.split("::")[-1]
        in_executor_class = test_id.startswith(EXECUTOR_CLASS)
        test_passes = passes.get(test_id)
        kept: list[tuple] = []
        understood = True
        for fixture, (lines, pairs) in edited.items():
            if in_executor_class:
                if fixture in TPC_QUERY_LISTS:
                    if method in TPC_QUERY_LISTS[fixture]:
                        kept += [("query", i) for i in sorted(pairs)]
                    continue  # the other executor tests only load the list in the class setup
                understood = False
                break
            if test_passes is None or not any(
                    label[0] in FIXTURE_LOADERS and label[1] == fixture for label in test_passes.labels):
                understood = False  # it reads the file some other way
                break
            kept += [("pairs", fixture, i) for i in sorted(pairs)] + [("lines", fixture, i) for i in sorted(lines)]
        if not understood:
            continue
        settled.add(test)
        if in_executor_class and method not in every_tpc_method:
            candidates.discard(test)
            continue
        if not test_id.startswith("tests/test_executor.py"):
            kept = [label for label in kept if label in test_passes.labels]
        if not kept:
            candidates.discard(test)
            continue
        reductions.stand_ins[test_id] = OnlyPasses(frozenset(kept), "query" if in_executor_class else "fixture")
        total = len(test_passes.labels) if test_passes is not None else 100
        reductions.reduced_cpu[test] = run_cpu[test] * min(1.0, len(kept) / max(total, 1))
    return settled


# ---- dialect families ---------------------------------------------------------------------
def is_dialect_module(module: str) -> bool:
    path = module.replace(".", "/") + ".py"
    return path.startswith(_DIALECT_DIRS) and path.rsplit("/", 1)[1][:-3] not in _SHARED_STEMS


def _top_level_objects(module, source: str) -> set[int]:
    """ids of the containers, classes and functions a module defines at its top level."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(x.id for target in node.targets for x in ast.walk(target) if isinstance(x, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    kinds = (dict, list, set, frozenset, type, types.FunctionType)
    return {id(value) for name, value in vars(module).items() if name in names and isinstance(value, kinds)}


def build_family_table(sqlglot_root: str) -> dict:
    """{dialect module path: (dialect names, module paths it reaches) or None}, for every
    dialect-specific module of the pinned tree. Imports everything: run it in a fresh process
    or a fork. The family of a module is the set of dialects whose classes (dialect,
    tokenizer, parser, generator and their bases), or the class attributes those hold, come
    from that module or from a dialect module that holds one of its objects. None when a
    module outside the dialect ones holds its objects, or when it reaches the default or the
    duckdb dialect (both are used everywhere)."""
    import importlib
    import sys

    root = os.path.realpath(sqlglot_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import sqlglot.dialects as registry
    from sqlglot.dialects import DIALECTS
    from sqlglot.dialects.dialect import Dialect

    for name in DIALECTS:
        getattr(registry, name)
    names = []
    for directory in _DIALECT_DIRS:
        for file_name in sorted(os.listdir(os.path.join(root, directory))):
            module = (directory + file_name[:-3]).replace("/", ".")
            if file_name.endswith(".py") and is_dialect_module(module):
                names.append(module)
    for name in names:
        importlib.import_module(name)
    owners: dict[int, set[str]] = {}  # object id -> dialect modules defining it (aliases: several)
    for name in names:
        module = sys.modules[name]
        with open(os.path.join(root, name.replace(".", "/") + ".py"), encoding="utf-8") as handle:
            source = handle.read()
        for ident in _top_level_objects(module, source):
            owners.setdefault(ident, set()).add(name)
        owners.setdefault(id(module), set()).add(name)
    holders: dict[str, set[str]] = {}  # dialect module -> modules holding one of its objects
    for name, module in list(sys.modules.items()):
        if module is None or name in _REGISTRY_MODULES or not (name == "sqlglot" or name.startswith("sqlglot.")):
            continue
        held = set()
        for value in vars(module).values():
            held |= owners.get(id(value), set())
        for owner in held - {name}:
            holders.setdefault(owner, set()).add(name)
    dialect_classes = {}
    for dialect_name, dialect in Dialect.classes.items():
        classes = []
        for klass in (dialect, getattr(dialect, "tokenizer_class", None), getattr(dialect, "parser_class", None),
                      getattr(dialect, "generator_class", None)):
            if isinstance(klass, type):
                classes += list(klass.__mro__)
        dialect_classes[dialect_name] = classes
    return {name.replace(".", "/") + ".py": _family(name, holders, owners, dialect_classes) for name in names}


def _family(start: str, holders, owners, dialect_classes):
    reached, todo = {start}, [start]
    while todo:
        for holder in holders.get(todo.pop(), ()):
            if holder in reached:
                continue
            if not is_dialect_module(holder):
                return None
            reached.add(holder)
            todo.append(holder)
    try:
        dialects = set()
        for dialect_name, classes in dialect_classes.items():
            if any(klass.__module__ in reached or any(owners.get(id(v), set()) & reached for v in vars(klass).values())
                   for klass in classes):
                dialects.add(dialect_name)
    except Exception:  # noqa
        return None
    if "" in dialects or "duckdb" in dialects:
        return None
    return dialects, {m.replace(".", "/") + ".py" for m in reached}


class DialectFamilies:
    def __init__(self, table_path: str, sqlglot_root: str, log):
        self.table_path = table_path
        self.sqlglot_root = sqlglot_root
        self.log = log
        self._table: dict | None = None
        self._by_patch: dict[tuple, tuple | None] = {}
        self._entry_dialects: dict[str, list | None] = {}

    def table(self) -> dict:
        if self._table is None and os.path.exists(self.table_path):
            from .atlas import read_artifact

            try:
                self._table = read_artifact(self.table_path)
            except Exception:  # noqa
                self._table = None
        if self._table is None:
            import time

            from .sandbox import in_child

            began = time.monotonic()
            self._table = in_child(lambda: build_family_table(self.sqlglot_root), 60) or {}
            self.log(f"dialect family table: {sum(v is not None for v in self._table.values())} of "
                     f"{len(self._table)} modules in {time.monotonic() - began:.1f}s")
        return self._table

    def of_patch(self, impact: Impact, patch: str):
        """(dialect names, module paths) for a patch confined to dialect-specific modules; None
        when it, or a module holding its objects, reaches beyond them, or when a class
        statement line changes (the class tree would differ)."""
        paths = impact.new_contents
        if not paths or any(not (p.endswith(".py") and is_dialect_module(p[:-3].replace("/", "."))) for p in paths):
            return None
        if _CLASS_LINE_EDIT.search(patch) or any(text is None for text in paths.values()):
            return None
        key = tuple(sorted(paths))
        if key not in self._by_patch:
            table = self.table()
            found = None
            if all(table.get(path) is not None for path in paths):
                dialects, modules = set(), set()
                for path in paths:
                    dialects |= table[path][0]
                    modules |= table[path][1]
                found = (dialects, modules)
            self._by_patch[key] = found
        return self._by_patch[key]

    def entry_dialects(self, analyzer: ImpactAnalyzer, fixture: str):
        """Per (sql, expected) pair of a .sql fixture: the dialects its leading ``# dialect:``
        meta line names ({""}: the default dialect). None when a meta line can't be read."""
        if fixture not in self._entry_dialects:
            text = analyzer.source("tests/fixtures/" + fixture)
            self._entry_dialects[fixture] = None if text is None else _dialects_per_pair(text)
        return self._entry_dialects[fixture]


def _dialects_per_pair(text: str) -> list[set[str]] | None:
    found = []
    for sql, _ in fixture_entries(text)[1]:
        meta = {}
        for line in sql.split("\n"):
            if not line.startswith("#"):
                break  # (tests/helpers.py reads the leading '#' lines only)
            if ":" not in line:
                return None
            key, value = line.split(":", 1)
            meta[key.lstrip("#").strip()] = value.strip()
        named = meta.get("dialect") or ""
        found.append(set(named.split(", ")) if named else {""})
    return found


def narrow_to_dialects(families: DialectFamilies, analyzer: ImpactAnalyzer, impact: Impact, patch: str,
                       tests_by_id: dict[str, int], candidates: set[int], data_readers: set[int],
                       passes: dict[str, TestPasses], run_cpu: list[float], dialect_base_s: float,
                       reductions: Reductions) -> int:
    """A class/module-level edit confined to some dialects' modules: a fixture-driven test runs
    the entries written for one of those dialects, and the entries whose recorded run-time
    lines reach an affected module (the rest of its method runs as usual). Returns how many
    tests it reduced."""
    family = families.of_patch(impact, patch)
    if family is None:
        return 0
    dialects, modules = family
    import_time_lines = analyzer.atlas.import_time_lines
    reduced = 0
    for test_id, test_passes in passes.items():
        test = tests_by_id.get(test_id)
        if (test is None or test not in candidates or test in data_readers or test_id in reductions.stand_ins
                or test_id in LOOPING_TESTS):
            continue
        labels = test_passes.labels
        if len(labels) < 5 or not all(label[0] in FIXTURE_LOADERS for label in labels):
            continue
        # lines that every entry ran come from the method around the loop, which always runs;
        # import-time lines go to whichever entry first loaded a dialect
        every_pass = test_passes.everyone
        reach_lines = {path: {line for line, hit in test_passes.reach.get(path, {}).items()
                              if hit != every_pass and line not in import_time_lines.get(path, ())}
                       for path in modules}
        kept = set(test_passes.reaching(reach_lines))
        readable = True
        for label in labels:
            if label[0] != "pairs":
                continue  # (line fixtures carry no meta: the default dialect)
            per_pair = families.entry_dialects(analyzer, label[1])
            if per_pair is None or label[2] >= len(per_pair):
                readable = False
                break
            if per_pair[label[2]] & dialects:
                kept.add(label)
        if not readable:
            continue
        reductions.stand_ins[test_id] = OnlyPasses(frozenset(kept), pass_kind(labels))
        reductions.dialect_reduced.add(test)
        reductions.reduced_cpu[test] = run_cpu[test] * (len(kept) + 1) / (len(labels) + 1) + dialect_base_s
        reduced += 1
    return reduced
