"""Loading the precomputed knowledge about the pinned tree (the "atlas", built by ``forge``).

Every atlas file is a gzip'd pickle of plain containers. They are read with an unpickler
that refuses to build anything but builtin containers, so a damaged or foreign file can't
run code.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
from dataclasses import dataclass

from . import bits


class _PlainDataUnpickler(pickle.Unpickler):
    _ALLOWED = {("builtins", "set"), ("builtins", "frozenset")}

    def find_class(self, module: str, name: str):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"atlas files hold plain data only, not {module}.{name}")


def read_artifact(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as handle:
        return _PlainDataUnpickler(handle).load()


def write_artifact(path: str, payload, level: int = 6) -> None:
    temporary = path + ".partial"
    with gzip.open(temporary, "wb", compresslevel=level) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def module_name(path: str) -> str:
    """sqlglot/optimizer/__init__.py -> sqlglot.optimizer"""
    stem = path[:-3] if path.endswith(".py") else path
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def test_module(test_id: str) -> str:
    """tests/test_x.py::Class::method -> tests.test_x"""
    return test_id.split("::", 1)[0][:-3].replace("/", ".")


def test_class(test_id: str) -> str:
    return test_id.rsplit("::", 1)[0]


class CoverageAtlas:
    """What each test of the pinned tree executed, read, loaded and instantiated, plus the
    static import graph and the read sites of class/module-level names."""

    def __init__(self, payload: dict, test_ids: list[str]):
        self.size = len(test_ids)
        self.everyone = bits.full(self.size)
        translate, self.unrecorded = self._translator(payload["test_ids"], test_ids)

        self.line_hits: dict[str, dict[int, int]] = {
            path: {line: translate(tests) for line, tests in hits.items()}
            for path, hits in payload["line_hits"].items()}
        self.file_reads: dict[str, int] = {p: translate(t) for p, t in payload["file_reads"].items()}
        self.dir_scans: dict[str, int] = {p: translate(t) for p, t in payload.get("dir_scans", {}).items()}
        self.instantiated: dict[str, int] = {c: translate(t) for c, t in payload["instantiated"].items()}
        self.module_loads: dict[str, int] = {m: translate(t) for m, t in payload.get("module_loads", {}).items()}
        self.import_time_lines: dict[str, set[int]] = {p: set(ls) for p, ls in payload["import_time_lines"].items()}
        # path -> [(first line, last line, {(body path, body qualname, body line), ...})]
        self.import_time_calls: dict[str, list] = payload.get("import_time_calls", {})
        self.imported_by: dict[str, list[str]] = payload.get("imported_by", {})
        self.lazy_imports: dict[str, list] = payload.get("lazy_imports", {})  # path -> [(importer, first, last)]
        self.name_reads: dict[str, list] = payload.get("name_reads", {})  # name -> [(path, first, last)]
        self.dynamic_imports: dict[str, list] = payload.get("dynamic_imports", {})  # prefix -> [(path, first, last)]

        self.file_hits = {path: _union(hits.values()) for path, hits in self.line_hits.items()}
        self.classes_by_module: dict[str, dict[str, int]] = {}
        self.expression_users: dict[str, int] = {}
        for key, tests in self.instantiated.items():
            module, qualname = key.split(":", 1)
            self.classes_by_module.setdefault(module, {})[qualname] = tests
            if module.startswith("sqlglot.expressions"):
                self.expression_users[qualname] = self.expression_users.get(qualname, 0) | tests

    def _translator(self, recorded_ids: list[str], test_ids: list[str]):
        """Bit positions follow the recorded test list; when today's list differs, rewrite
        them to today's positions. Today's tests the atlas never saw are reported apart."""
        if recorded_ids == test_ids:
            return (lambda tests: tests), 0
        position = {test_id: index for index, test_id in enumerate(test_ids)}
        moved = [position.get(test_id) for test_id in recorded_ids]
        seen = bits.from_indices(index for index in moved if index is not None)

        def translate(tests: int) -> int:
            return bits.from_indices(moved[i] for i in bits.members(tests) if moved[i] is not None)

        return translate, self.everyone & ~seen

    def tests_on_lines(self, path: str, first: int, last: int) -> int:
        hits = self.line_hits.get(path)
        if not hits:
            return 0
        tests = 0
        for line in range(first, last + 1):
            tests |= hits.get(line, 0)
        return tests

    def import_time_bodies(self, path: str, line: int):
        """For the innermost function containing ``line`` that ran during import: the class or
        module body lines it ran for. None when no such call was traced."""
        best = None
        for first, last, bodies in self.import_time_calls.get(path, ()):
            if first <= line <= last and (best is None or last - first < best[1] - best[0]):
                best = (first, last, bodies)
        return best[2] if best else None

    def ran_at_import(self, path: str, first: int, last: int) -> bool:
        lines = self.import_time_lines.get(path, ())
        return any(line in lines for line in range(first, last + 1))


def _union(values) -> int:
    total = 0
    for value in values:
        total |= value
    return total


@dataclass
class TestPasses:
    """A slow test made of independent passes (dialects, queries, fixture entries), with the
    sqlglot lines each pass ran: ``reach[path][line]`` is a bitset over ``labels``."""

    labels: list[tuple]
    reach: dict[str, dict[int, int]]

    def reaching(self, edited: dict[str, set[int]]) -> list[tuple]:
        """Labels of the passes that run any of the edited lines."""
        chosen = 0
        for path, lines in edited.items():
            hits = self.reach.get(path)
            if hits:
                for line in lines:
                    chosen |= hits.get(line, 0)
        return [self.labels[i] for i in bits.members(chosen)]

    @property
    def everyone(self) -> int:
        return bits.full(len(self.labels))


def load_passes(path: str, test_ids: list[str]) -> dict[str, TestPasses]:
    known = set(test_ids)
    return {test_id: TestPasses(entry["labels"], entry["reach"])
            for test_id, entry in read_artifact(path).items()
            if test_id in known and entry is not None}


def load_costs(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return {}
