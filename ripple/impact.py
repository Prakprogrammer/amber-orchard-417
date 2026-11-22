"""Which tests can a patch affect? Coverage atlas + static scopes -> a bitset of candidates.

Rules, per edited file:
- data file (fixture): the tests that opened it ("data readers": they nearly always fail);
- a single comparison operator or literal operand: the tests whose recorded evaluations of
  that comparison flip (``ripple.compare``);
- an edit inside a function body: the tests that ran the function (slow tests only when they
  ran the edited statement itself);
- a class-level edit: the tests that created the class, narrowed to the readers of the edited
  attribute or the users of the expression classes a table entry names, when provable;
- a module-level edit: the tests that load the module;
- a function body that ran during import: whatever the class/module bodies it ran for built;
- cosmetic edits (same AST): nothing;
- anything else it can't reason about: every test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import bits
from .atlas import CoverageAtlas, module_name
from .compare import ComparisonAtlas, find_single_comparison_edit
from .diffs import FileDiff, added_new_lines, patch_text, touched_old_lines
from .syntax import ClassScope, FunctionScope, Outline, could_be_cosmetic, outline, same_tree

SLOW_TEST_SECONDS = 2.0
_PYTEST_CONFIG = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "conftest.py")


@dataclass
class EditedTree:
    old: object
    new: object
    removed: set[int]  # old-file lines removed
    added: set[int]  # new-file lines added


@dataclass
class Impact:
    candidates: int = 0
    data_readers: int = 0  # tests that read an edited data file
    python_changed: bool = False
    sqlglot_python_changed: bool = False
    everything: bool = False  # no reasoning possible: every test is a candidate
    unchanged: bool = True  # the patch changes no file's contents
    certain: bool = False  # a recorded comparison flips: the candidates surely behave differently
    reaches_import_time: bool = False  # an edited function body ran while sqlglot was imported
    new_contents: dict[str, str | None] = field(default_factory=dict)  # path -> text (None: gone)
    trees: dict[str, EditedTree] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def decided(self) -> bool:
        """A normal selection: neither "everything" nor "nothing changed"."""
        return not self.everything and not self.unchanged


class ImpactAnalyzer:
    def __init__(self, atlas: CoverageAtlas, tests: list[dict], sqlglot_root: str,
                 comparisons: ComparisonAtlas | None = None):
        self.atlas = atlas
        self.comparisons = comparisons
        self.root = sqlglot_root
        self.slow = bits.from_indices(i for i, t in enumerate(tests) if t["seconds"] >= SLOW_TEST_SECONDS)
        self.by_test_file: dict[str, int] = {}
        for index, test in enumerate(tests):
            self.by_test_file[test["file"]] = self.by_test_file.get(test["file"], 0) | (1 << index)
        self._sources: dict[str, str | None] = {}
        self._outlines: dict[str, Outline] = {}
        self._module_users: dict[str, int] = {}
        self._body_users: dict[tuple[str, str], int] = {}

    # ---- the pinned tree -------------------------------------------------------------
    def source(self, path: str) -> str | None:
        if path not in self._sources:
            try:
                with open(os.path.join(self.root, path), encoding="utf-8", newline="") as handle:
                    self._sources[path] = handle.read()
            except (OSError, UnicodeDecodeError):
                self._sources[path] = None
        return self._sources[path]

    def pinned_outline(self, path: str, text: str) -> Outline:
        if path not in self._outlines:
            self._outlines[path] = outline(text)
        return self._outlines[path]

    # ---- who uses what -------------------------------------------------------------
    def test_file_tests(self, path: str) -> int:
        """The tests in a test file and in every test file importing it, transitively."""
        tests, seen, todo = 0, set(), [path]
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            tests |= self.by_test_file.get(current, 0)
            todo.extend(self.atlas.imported_by.get(current, []))
        return tests

    def class_users(self, module: str, qualname: str) -> int:
        return self.atlas.classes_by_module.get(module, {}).get(qualname, 0)

    def _classes_of_module(self, module: str) -> int:
        tests = 0
        for users in self.atlas.classes_by_module.get(module, {}).values():
            tests |= users
        return tests

    def module_users(self, path: str) -> int:
        """Tests that loaded the module when run alone in a fresh process (so a module-level
        edit, or an import failure, can reach them), plus its runtime and class users."""
        if not self.atlas.module_loads or path.startswith("tests/"):
            return self._module_users_by_graph(path)
        if path not in self._module_users:
            atlas = self.atlas
            self._module_users[path] = (atlas.module_loads.get(module_name(path), 0) | atlas.file_hits.get(path, 0)
                                        | atlas.file_reads.get(path, 0) | self._classes_of_module(module_name(path)))
        return self._module_users[path]

    def _module_users_by_graph(self, path: str) -> int:
        """Runtime users of the module and of every module importing it, transitively
        (module-level values travel through ``from x import NAME`` chains)."""
        if path in self._module_users:
            return self._module_users[path]
        atlas, tests, seen, todo = self.atlas, 0, set(), [path]
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            if current.startswith("tests/"):
                tests |= self.test_file_tests(current)
                continue
            module = module_name(current)
            tests |= atlas.file_hits.get(current, 0) | atlas.file_reads.get(current, 0) | self._classes_of_module(module)
            for importer, first, last in atlas.lazy_imports.get(current, ()):
                tests |= atlas.tests_on_lines(importer, first, last)
            for prefix, sites in atlas.dynamic_imports.items():
                if module.startswith(prefix):
                    for site_path, first, last in sites:
                        # a fully computed name inside sqlglot (serde rebuilding classes by name)
                        # only loads modules whose classes the test then creates: covered above
                        if prefix or site_path.startswith("tests/"):
                            tests |= atlas.tests_on_lines(site_path, first, last)
            todo.extend(atlas.imported_by.get(current, []))
        self._module_users[path] = tests
        return tests

    def readers_in_test_code(self, name: str) -> int:
        """Tests whose own code reads ``name`` (for instance, one that checks a whole table)."""
        tests = 0
        for path, first, last in self.atlas.name_reads.get(name, ()):
            if path.startswith("tests/"):
                tests |= self.atlas.tests_on_lines(path, first, last)
        return tests

    def readers(self, name: str) -> int | None:
        """Tests that ran a statement reading ``name``; None when unknown or read at import."""
        sites = self.atlas.name_reads.get(name)
        if not sites:
            return None
        tests = 0
        for path, first, last in sites:
            if self.atlas.ran_at_import(path, first, last):
                return None
            tests |= self.atlas.tests_on_lines(path, first, last)
        return tests

    def readers_of_all(self, names) -> int | None:
        tests = 0
        for name in names:
            found = self.readers(name)
            if found is None:
                return None
            tests |= found
        return tests

    def expression_users(self, names) -> int:
        """Tests that created one of the named expression classes. A name that is no known
        expression class (an enum member, a token type...) says nothing: every test."""
        tests = 0
        for name in names:
            users = self.atlas.expression_users.get(name)
            if not users:
                return self.atlas.everyone
            tests |= users
        return tests

    def _body_tests(self, path: str, body: str) -> int:
        """Tests reached by what the class/module body ``body`` of ``path`` built at import."""
        key = (path, body)
        if key not in self._body_users:
            if path.startswith("tests/"):
                tests = self.test_file_tests(path)
            elif body == "<module>":
                tests = self.module_users(path)
            else:
                tests = 0
                for qualname, users in self.atlas.classes_by_module.get(module_name(path), {}).items():
                    if qualname == body or qualname.startswith(body + "."):
                        tests |= users
                text = self.source(path)
                scope = self.pinned_outline(path, text).klass(body) if text is not None else None
                if scope is not None:
                    tests |= self.atlas.tests_on_lines(path, scope.first_line, scope.last_line)
                if tests == 0:
                    tests = self.module_users(path)
            self._body_users[key] = tests
        return self._body_users[key]

    def _body_tests_narrowed(self, path: str, body: str, body_lines: set) -> int:
        """``_body_tests``, narrowed as a class/module-level edit at the body lines that made
        the import-time call would be."""
        tests = self._body_tests(path, body)
        if path.startswith("tests/") or None in body_lines:
            return tests
        text = self.source(path)
        if text is None:
            return tests
        pinned = self.pinned_outline(path, text)
        classes, names = set(), set()
        classes_known = names_known = True
        for line in body_lines:
            named = pinned.table_entry_classes(line)
            name = pinned.assigned_name(line)
            if named is None:
                classes_known = False
            else:
                classes |= named
            if name is None:
                names_known = False
            else:
                names.add(name)
        if classes_known and classes:
            tests &= self.expression_users(classes)
        if names_known and names:
            readers = self.readers_of_all(names)
            if readers is not None:
                tests &= readers
        for name in names:
            tests |= self.readers_in_test_code(name)
        return tests

    # ---- the patch ---------------------------------------------------------------------
    def analyze(self, diffs: list[FileDiff]) -> Impact:
        impact = Impact()
        for diff in diffs:
            before_path, after_path = diff.source, diff.target
            path = before_path or after_path or ""
            before = self.source(before_path) if before_path else None
            try:
                after = None if diff.is_binary else patch_text(before, diff)
            except Exception as problem:  # a patch that does not apply: stay safe
                impact.everything = True
                impact.notes.append(f"{path}: does not apply ({problem})")
                continue
            if before_path and after_path and before_path != after_path:
                impact.new_contents[before_path] = None
            if after_path:
                impact.new_contents[after_path] = after
            elif before_path:
                impact.new_contents[before_path] = None
            if before == after and before_path == after_path:
                continue
            impact.unchanged = False
            _FileEdit(self, impact, diff, path, before, after).judge()
        return impact


class _FileEdit:
    """The judgement of one edited file of a patch."""

    def __init__(self, analyzer: ImpactAnalyzer, impact: Impact, diff: FileDiff, path: str,
                 before: str | None, after: str | None):
        self.analyzer = analyzer
        self.atlas = analyzer.atlas
        self.impact = impact
        self.diff = diff
        self.path = path
        self.before = before
        self.after = after
        self.in_sqlglot = path.startswith("sqlglot/")
        self.in_tests = path.startswith("tests/")
        self.renamed_paths = {diff.source, diff.target} - {None}
        # gathered while walking the edited lines
        self.function_tests = 0
        self.class_edits: dict[str, set | None] = {}  # class -> named expression classes (None: unknown)
        self.class_names: dict[str, set | None] = {}  # class -> assigned names (None: unknown)
        self.module_edits: list[frozenset | None] = []
        self.module_names: list[str | None] = []
        self.extra = 0
        self.notes: list[str] = []

    def _add(self, tests: int, note: str, data: bool = False) -> None:
        self.impact.candidates |= tests
        if data:
            self.impact.data_readers |= tests
        if note:
            self.impact.notes.append(f"{self.path}: {note}")

    def judge(self) -> None:
        if not self.path.endswith(".py"):
            return self._data_file()
        if not (self.in_sqlglot or self.in_tests):
            return self._outside_file()
        self.impact.python_changed = True
        if self.in_sqlglot:
            self.impact.sqlglot_python_changed = True
        if self.before is None or self.after is None or self.diff.source != self.diff.target:
            return self._added_removed_or_renamed()
        self._add(self.atlas.file_reads.get(self.path, 0), "")
        if could_be_cosmetic(self.diff.rows_marked("-"), self.diff.rows_marked("+")) and same_tree(self.before, self.after):
            return self._add(0, "AST unchanged (comments/format)")
        self.old = self.analyzer.pinned_outline(self.path, self.before)
        self.new = outline(self.after)
        if self.new.broken:
            return self._syntax_error()
        self.rewritten, self.borders = touched_old_lines(self.diff)
        self.added = added_new_lines(self.diff)
        self.impact.trees[self.path] = EditedTree(self.old.tree, self.new.tree, self.rewritten, self.added)
        if self.analyzer.comparisons is not None and self.in_sqlglot and self._comparison_edit():
            return
        self._code_edit()

    # ---- whole-file cases ----------------------------------------------------------
    def _data_file(self) -> None:
        tests = 0
        for path in self.renamed_paths:
            tests |= self.atlas.file_reads.get(path, 0)
        if self.in_tests and os.path.basename(self.path) in ("conftest.py", "pytest.ini"):
            self.impact.everything = True
        if self.path in _PYTEST_CONFIG:
            self.impact.notes.append(f"{self.path}: pytest config")
            self.impact.everything = True
        self._add(tests, f"data file -> {bits.count(tests)} tests", data=True)

    def _outside_file(self) -> None:
        if os.path.basename(self.path) == "conftest.py":
            self.impact.everything = True
            return self._add(0, "conftest")
        # for instance sqlglotc/setup.py, which a test runs through runpy
        tests = self.atlas.file_hits.get(self.path, 0) | self.atlas.file_reads.get(self.path, 0)
        for path in self.renamed_paths:
            tests |= self.atlas.dir_scans.get(os.path.dirname(path), 0)
        self._add(tests, f"outside sqlglot/tests -> {bits.count(tests)}")

    def _added_removed_or_renamed(self) -> None:
        # its users, plus the tests that list its directory (pkgutil.iter_modules and alike)
        tests = 0
        for path in self.renamed_paths:
            tests |= self.atlas.dir_scans.get(os.path.dirname(path), 0)
        if self.before is not None:
            tests |= self.analyzer.module_users(self.diff.source)
            if self.in_tests:
                tests |= self.analyzer.test_file_tests(self.diff.source)
        if self.after is not None and outline(self.after).broken:
            tests |= self.analyzer.module_users(self.diff.target)
        self._add(tests, f"added/removed/renamed -> {bits.count(tests)}")

    def _syntax_error(self) -> None:
        tests = self.analyzer.module_users(self.path) | (self.analyzer.test_file_tests(self.path) if self.in_tests else 0)
        if self.in_sqlglot and (self.path.count("/") == 1 or self.path.endswith("__init__.py")):
            self.impact.everything = True
        self._add(tests, f"syntax error -> {bits.count(tests)}")

    def _comparison_edit(self) -> bool:
        edit = find_single_comparison_edit(self.old.tree, self.new.tree, self.rewritten, self.added)
        if edit is None:
            return False
        flipped = self.analyzer.comparisons.tests_flipped_by(self.path, edit)
        if flipped is None:
            return False
        line, _, end_line, _ = edit.where
        unsure = self.atlas.tests_on_lines(self.path, line, end_line) & self.analyzer.comparisons.untrusted
        tests = flipped | unsure
        self._add(tests, f"compare edit line {line} {edit.describe()} -> {bits.count(tests)}")
        if flipped:
            self.impact.certain = True
        return True

    # ---- line by line ----------------------------------------------------------------
    def _code_edit(self) -> None:
        for line in sorted(self.rewritten):
            self._place(self.old, line)
        for line in sorted(self.added):
            self._place(self.new, line)
        self._import_time_reach()
        self._narrow_slow_tests()
        self.extra |= self.function_tests
        self._class_level()
        self._module_level()
        self.impact.candidates |= self.extra
        self.impact.notes.append(f"{self.path}: " + ", ".join(self.notes))

    def _place(self, side: Outline, line: int) -> None:
        scope = side.scope_at(line)
        if not isinstance(scope, FunctionScope):
            return self._namespace_edit(side, line, scope)
        # the name and signature of a method are part of its class namespace too
        if (line < scope.body_line or line == scope.def_line) and scope.owner_class is not None:
            owner = side.klass(scope.owner_class)
            if owner is not None:
                self._namespace_edit(side, line, owner)
        if self._function_edit(scope, side):
            return
        # a function with no pinned counterpart: an edit of the namespace that defines it
        owner = side.klass(scope.owner_class) if scope.owner_class is not None else None
        if owner is not None:
            self._namespace_edit(side, line, owner)
        elif "<locals>" in scope.qualname:
            outer = self.old.function(scope.qualname.split(".<locals>.")[0])
            if outer is not None:
                self._function_edit(outer, self.old)
            else:
                self._unknown_module_edit()
        else:
            self._unknown_module_edit()

    def _function_edit(self, scope: FunctionScope, side: Outline) -> bool:
        if side is self.new:
            if scope.is_lambda:
                return False  # lambdas have no unique name to match: a new definition
            scope = self.old.function(scope.qualname)
            if scope is None:
                return False  # a new function
        tests = self.atlas.tests_on_lines(self.path, scope.first_line, scope.last_line)
        self.function_tests |= tests
        self.notes.append(f"func {scope.qualname}:{bits.count(tests)}")
        return True

    def _unknown_module_edit(self) -> None:
        self.module_edits.append(None)
        self.module_names.append(None)

    def _namespace_edit(self, side: Outline, line: int, scope: ClassScope | None) -> None:
        named = side.table_entry_classes(line)
        name = side.assigned_name(line)
        if scope is None:
            self.module_edits.append(named)
            self.module_names.append(name)
            return
        known = self.class_edits.get(scope.qualname, set())
        self.class_edits[scope.qualname] = None if known is None or named is None else known | set(named)
        names = self.class_names.get(scope.qualname, set())
        self.class_names[scope.qualname] = None if names is None or name is None else names | {name}

    def _import_time_reach(self) -> None:
        """An edited statement in a function that ran during import (metaclasses, table
        builders...) reaches what the class/module bodies it ran for built. Untraced: all."""
        if not self.atlas.import_time_lines.get(self.path):
            return
        for line in sorted(self.rewritten | self.borders):
            scope = self.old.scope_at(line)
            if (not isinstance(scope, FunctionScope) or scope.is_lambda or line < scope.body_line
                    or line == scope.def_line):
                continue  # def and signature lines: the namespace rules handle them
            statement = self.old.statement_lines(line)
            if statement is None:
                continue
            first, last = statement
            ran = [x for x in range(first, last + 1) if x in self.atlas.import_time_lines[self.path]]
            if not ran:
                continue
            bodies, untraced = set(), False
            for at in ran:
                found = self.atlas.import_time_bodies(self.path, at)
                if found is None:
                    untraced = True
                else:
                    bodies.update(found)
            if untraced or not bodies:
                self.notes.append(f"{scope.qualname} line {line} runs at import (untraced)")
                self.impact.everything = True
                break
            by_body: dict[tuple[str, str], set] = {}
            for body in bodies:
                by_body.setdefault((body[0], body[1]), set()).add(body[2] if len(body) > 2 else None)
            reached = 0
            for (body_path, body_name), body_lines in by_body.items():
                reached |= self.analyzer._body_tests_narrowed(body_path, body_name, body_lines)
            self.extra |= reached
            self.impact.reaches_import_time = True
            self.notes.append(f"{scope.qualname} line {line} runs at import for {len(bodies)} bodies:{bits.count(reached)}")

    def _narrow_slow_tests(self) -> None:
        """Slow tests stay candidates only if they ran an edited statement itself (when every
        edit is a rewrite of existing statements inside function bodies)."""
        if self.diff.added_count > self.diff.removed_count:
            return
        for line in self.rewritten:
            scope = self.old.scope_at(line)
            if not isinstance(scope, FunctionScope) or line < scope.body_line:
                return
        if not self.function_tests & self.analyzer.slow:
            return
        ran_edit = 0
        for line in self.rewritten | self.borders:
            first, last = self.old.statement_lines(line) or (line, line)
            ran_edit |= self.atlas.tests_on_lines(self.path, first, last)
        dropped = self.function_tests & self.analyzer.slow & ~ran_edit
        if dropped:
            self.notes.append(f"heavy dropped {bits.count(dropped)}")
        self.function_tests &= ~dropped

    def _class_level(self) -> None:
        analyzer, module = self.analyzer, module_name(self.path)
        for qualname, named in self.class_edits.items():
            tests = analyzer.class_users(module, qualname)
            scope = self.old.klass(qualname) or self.new.klass(qualname)
            if scope is not None:
                tests |= self.atlas.tests_on_lines(self.path, scope.first_line, scope.last_line)
            if self.in_tests:
                tests |= analyzer.test_file_tests(self.path)
            if tests == 0:
                tests = analyzer.module_users(self.path)
            if named:
                tests &= analyzer.expression_users(named)
            names = self.class_names.get(qualname)
            if names and not self.in_tests:
                readers = analyzer.readers_of_all(names)
                if readers is not None:
                    tests &= readers
                    self.notes.append(f"names {','.join(sorted(names))}")
            for name in names or ():
                tests |= analyzer.readers_in_test_code(name)
            self.extra |= tests
            label = f"[{','.join(sorted(named))}]" if named else ""
            self.notes.append(f"class {qualname}{label}:{bits.count(tests)}")

    def _module_level(self) -> None:
        if not self.module_edits:
            return
        analyzer = self.analyzer
        tests = analyzer.module_users(self.path)
        if self.in_tests:
            tests |= analyzer.test_file_tests(self.path)
        if all(self.module_edits):
            named = set().union(*self.module_edits)
            tests &= analyzer.expression_users(named)
            self.notes.append(f"module[{','.join(sorted(named))}]:{bits.count(tests)}")
        else:
            self.notes.append(f"module:{bits.count(tests)}")
        names = set(self.module_names)
        if all(self.module_names) and not self.in_tests:
            readers = analyzer.readers_of_all(names)
            if readers is not None:
                tests &= readers
                self.notes.append(f"names {','.join(sorted(names))}:{bits.count(tests)}")
        for name in names - {None}:
            tests |= analyzer.readers_in_test_code(name)
        self.extra |= tests
