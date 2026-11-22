"""Applying a function-body-only patch without importing anything again.

The parent imports sqlglot and the test modules once. For a patch that only edits function
bodies, the child swaps the patched code objects into the live function objects of those
modules. That is only exact when
- module and class bodies compile to the same code (defaults, decorators and class attributes
  are built there),
- the function never ran in the parent (results it produced could sit cached in module
  state), unless only functions nested inside it changed,
- the free and cell variables are the same (closures stay valid), and docstrings don't change.
Changed modules the parent never imported are served patched by an import hook instead.
"""

from __future__ import annotations

import ast
import os
import sys
import types
from dataclasses import dataclass

from .syntax import edited_functions

_CO_OPTIMIZED = 0x0001  # set on functions, not on module or class bodies
_MONITOR_TOOL = 5


def _shape(code: types.CodeType):
    """A code object, ignoring line numbers and the code of the functions it defines."""

    def normal(value):
        if isinstance(value, types.CodeType):
            return ("<code>", value.co_qualname)
        if isinstance(value, tuple):
            return ("tuple", tuple(normal(v) for v in value))
        if isinstance(value, frozenset):
            return ("frozenset", frozenset(normal(v) for v in value))
        return (type(value).__name__, value)

    return (code.co_code, normal(code.co_consts), code.co_names, code.co_varnames, code.co_freevars,
            code.co_cellvars, code.co_flags, code.co_argcount, code.co_posonlyargcount,
            code.co_kwonlyargcount, code.co_exceptiontable)


def _nested_code(code: types.CodeType) -> dict[tuple[str, int], types.CodeType]:
    """Every code object defined (at any depth) in ``code``, keyed by (qualname, occurrence)."""
    found: dict[tuple[str, int], types.CodeType] = {}
    seen: dict[str, int] = {}

    def visit(outer):
        for constant in outer.co_consts:
            if isinstance(constant, types.CodeType):
                occurrence = seen.get(constant.co_qualname, 0)
                seen[constant.co_qualname] = occurrence + 1
                found[(constant.co_qualname, occurrence)] = constant
                visit(constant)

    visit(code)
    return found


def could_swap(new_contents: dict[str, str | None], import_time_lines: dict, trees: dict) -> bool:
    """A quick static guess, before anything is imported: every difference lies inside a
    function body, and no such body ran while sqlglot was imported."""
    for path, text in new_contents.items():
        if text is None or not path.startswith("sqlglot/") or not path.endswith(".py"):
            return False
        edited = trees.get(path)
        if edited is None or edited.old is None or edited.new is None:
            return False
        functions = edited_functions(edited.old, edited.new, edited.removed, edited.added)
        if not functions:
            return False
        ran = import_time_lines.get(path, ())
        for function in functions:
            body = function.body if isinstance(function.body, list) else [function.body]
            first, last = body[0].lineno, function.end_lineno or body[0].lineno
            if isinstance(function, ast.Lambda):
                first = last + 1  # its line also holds its definition: the child's check decides
            if any(first <= line <= last for line in ran):
                return False
    return True


class _PatchedSourceLoader:
    def __init__(self, file_name: str, source: str):
        self.file_name, self.source = file_name, source

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.source, self.file_name, "exec", dont_inherit=True), module.__dict__)

    def get_source(self, fullname):
        return self.source


class _PatchedModuleFinder:
    def __init__(self, unloaded: dict[str, tuple[str, str]]):
        self.unloaded = unloaded

    def find_spec(self, fullname, path=None, target=None):
        import importlib.util

        if fullname not in self.unloaded:
            return None
        file_name, source = self.unloaded[fullname]
        package_dir = [os.path.dirname(file_name)] if file_name.endswith("__init__.py") else None
        return importlib.util.spec_from_file_location(fullname, file_name, loader=_PatchedSourceLoader(file_name, source),
                                                      submodule_search_locations=package_dir)


@dataclass
class Swap:
    replacements: list  # [(live function, patched code)]
    unloaded: dict  # module -> (file, patched source): imported patched on demand

    def apply(self) -> None:
        for function, code in self.replacements:
            function.__code__ = code
        if self.unloaded:
            sys.meta_path.insert(0, _PatchedModuleFinder(self.unloaded))


class Preload:
    """The parent's one-time import of sqlglot and the test modules, noting every sqlglot
    function that runs meanwhile (sys.monitoring, PY_START)."""

    def __init__(self, pinned_root: str):
        self.pinned_root = pinned_root
        self.package_dir = os.path.join(pinned_root, "sqlglot") + os.sep
        self.ran: set[tuple[str, str, int]] = set()  # (file, qualname, first line)
        self._live: dict | None = None
        self._pinned_code: dict[str, types.CodeType] = {}
        self._edits: dict[tuple[str, str], list | None] = {}

    def import_modules(self, names, log) -> None:
        import importlib

        monitoring = sys.monitoring

        def on_start(code, offset):
            if code.co_filename.startswith(self.package_dir):
                self.ran.add((code.co_filename, code.co_qualname, code.co_firstlineno))
            return monitoring.DISABLE

        try:
            monitoring.use_tool_id(_MONITOR_TOOL, "ripple-preload")
            monitoring.register_callback(_MONITOR_TOOL, monitoring.events.PY_START, on_start)
            monitoring.set_events(_MONITOR_TOOL, monitoring.events.PY_START)
            watching = True
        except ValueError:
            watching = False
        try:
            for name in names:
                try:
                    importlib.import_module(name)
                except BaseException as problem:  # noqa: a broken module is the child's problem
                    log(f"preload of {name} failed: {problem!r}")
        finally:
            if watching:
                monitoring.set_events(_MONITOR_TOOL, 0)
                monitoring.register_callback(_MONITOR_TOOL, monitoring.events.PY_START, None)
                monitoring.free_tool_id(_MONITOR_TOOL)

    def _live_functions(self) -> dict:
        if self._live is None:
            import gc

            self._live = {}
            for obj in gc.get_objects():
                if type(obj) is types.FunctionType and obj.__code__.co_filename.startswith(self.package_dir):
                    code = obj.__code__
                    self._live.setdefault((code.co_filename, code.co_qualname, code.co_firstlineno), []).append(obj)
        return self._live

    def _edited_code(self, path: str, source: str):
        """[(pinned code, patched code)] of the functions the patch edits, or None when anything
        else changes."""
        key = (path, source)
        if key not in self._edits:
            self._edits[key] = self._compare_compiled(path, source)
        return self._edits[key]

    def _compare_compiled(self, path: str, source: str):
        file_name = os.path.join(self.pinned_root, path)
        try:
            if file_name not in self._pinned_code:
                with open(file_name, encoding="utf-8", newline="") as handle:
                    self._pinned_code[file_name] = compile(handle.read(), file_name, "exec", dont_inherit=True)
            pinned = self._pinned_code[file_name]
            patched = compile(source, file_name, "exec", dont_inherit=True)
        except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
            return None
        if _shape(pinned) != _shape(patched):
            return None
        pinned_units, patched_units = _nested_code(pinned), _nested_code(patched)
        if set(pinned_units) != set(patched_units):
            return None
        edits = []
        for key, old in pinned_units.items():
            new = patched_units[key]
            if not (old.co_flags & _CO_OPTIMIZED and new.co_flags & _CO_OPTIMIZED):
                if _shape(old) != _shape(new):
                    return None  # a class body changed
                continue
            if old == new:
                continue
            if old.co_freevars != new.co_freevars or old.co_cellvars != new.co_cellvars:
                return None
            old_doc, new_doc = old.co_consts[:1], new.co_consts[:1]
            if old_doc != new_doc and any(isinstance(x, str) for x in old_doc + new_doc):
                return None  # a docstring: the live __doc__ would stay the pinned one
            edits.append((old, new))
        return edits

    def plan(self, new_contents: dict[str, str | None]) -> Swap | None:
        replacements, unloaded = [], {}
        for path, source in new_contents.items():
            if source is None or not path.startswith("sqlglot/") or not path.endswith(".py"):
                return None
            file_name = os.path.join(self.pinned_root, path)
            module = path[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            if module not in sys.modules:
                unloaded[module] = (file_name, source)
                continue
            if os.path.realpath(getattr(sys.modules[module], "__file__", "") or "") != os.path.realpath(file_name):
                return None
            edits = self._edited_code(path, source)
            if edits is None:
                return None
            for old, new in edits:
                key = (file_name, old.co_qualname, old.co_firstlineno)
                if key in self.ran and _shape(old) != _shape(new):
                    return None  # it ran in the parent: its results may be cached somewhere
                for function in self._live_functions().get(key, ()):
                    if function.__code__ != old:
                        return None
                    replacements.append((function, new))
        return Swap(replacements, unloaded)
