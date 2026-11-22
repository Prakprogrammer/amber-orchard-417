"""The comparison atlas: for every single comparison ``a OP b`` in sqlglot, what each test saw.

    python -m forge.comparisons record /data/tests.json <raw dir> [parallel]
    python -m forge.comparisons assemble <raw dir> /data/tests.json atlas/comparisons.pkl.gz

``record`` compiles every comparison of the pinned sqlglot through a recorder that evaluates the
real operator (same operands, order and exceptions) and notes, per evaluation, what the other
nine operators would have returned: a 20-bit word, two bits per operator (0 False, 1 True,
2 raised, 3 unknown). Alternatives are computed only where no Python-level ``__eq__``/``__lt__``
runs (builtin scalars, enums, tuples of those, objects with default comparisons): sqlglot's
Expression.__eq__ caches hashes on nodes, so calling it would change the run. Where one operand
is a literal, the other operand's values are kept too. Each test runs alone in a fresh fork;
evaluations during import mark their site import-time (never used by the selector).
"""

from __future__ import annotations

import ast
import builtins
import enum
import glob
import importlib.abc
import importlib.machinery
import json
import marshal
import os
import sys
import types

from forge import common
from ripple.compare import OPERATORS, RAISED, UNKNOWN, evaluate, is_literal

_NEGATION = {0: 1, 1: 0, 6: 7, 7: 6, 8: 9, 9: 8}
_ALL_UNKNOWN = sum(UNKNOWN << (2 * code) for code in range(10))
_SCALARS = {type(None), bool, int, float, str, bytes}
_COMPARISON_METHODS = ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__")
_BUILTIN_CALLABLES = (types.WrapperDescriptorType, types.MethodWrapperType, types.BuiltinFunctionType,
                      types.MethodDescriptorType)
SCALAR, ENUM, TUPLE, PLAIN_OBJECT, OPAQUE = range(5)

SITES: list[tuple] = []  # site number -> (path, line, column, end line, end column, operator, literal side)


class _Instrument(ast.NodeTransformer):
    def __init__(self, path: str):
        self.path = path

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        code = next(i for i, kind in enumerate(OPERATORS) if isinstance(node.ops[0], kind))
        side = "L" if is_literal(node.left) else "R" if is_literal(node.comparators[0]) else ""
        SITES.append((self.path, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, code, side))
        call = ast.Call(func=ast.Name(id="__forge_compare__", ctx=ast.Load()),
                        args=[ast.Constant(len(SITES) - 1), ast.Constant(code), node.left, node.comparators[0]],
                        keywords=[])
        return ast.copy_location(call, node)


class _InstrumentingLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        tree = _Instrument(path[len(common.PREFIX):]).visit(ast.parse(self.get_data(path), filename=path))
        return compile(ast.fix_missing_locations(tree), path, "exec", dont_inherit=True)


class _InstrumentingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader)
                and spec.origin and spec.origin.startswith(common.PREFIX + "sqlglot/")):
            spec.loader = _InstrumentingLoader(spec.loader.name, spec.loader.path)
        return spec


_kinds: dict[type, int] = {}


def _type_kind(kind: type) -> int:
    if kind in _kinds:
        return _kinds[kind]
    if kind in _SCALARS:
        found = SCALAR
    elif issubclass(kind, enum.Enum):
        custom = any(isinstance(getattr(kind, m, None), types.FunctionType) and getattr(kind, m).__module__ != "enum"
                     for m in _COMPARISON_METHODS)
        found = OPAQUE if custom else ENUM
    elif kind is tuple:
        found = TUPLE
    elif issubclass(kind, (list, dict, set, frozenset, tuple)):
        found = OPAQUE  # compares elements: may run anyone's __eq__
    else:
        custom = any(getattr(kind, m, None) is not None and not isinstance(getattr(kind, m), _BUILTIN_CALLABLES)
                     for m in _COMPARISON_METHODS)
        found = OPAQUE if custom else PLAIN_OBJECT
    _kinds[kind] = found
    return found


def _kind(value, depth: int = 0) -> int:
    found = _type_kind(type(value))
    if found == TUPLE and (depth > 2 or len(value) > 16 or any(_kind(v, depth + 1) not in (SCALAR, ENUM) for v in value)):
        return OPAQUE
    return found


def _keepable(value, depth: int = 0) -> bool:
    if type(value) in _SCALARS:
        return True
    return type(value) is tuple and depth < 3 and len(value) <= 16 and all(_keepable(v, depth + 1) for v in value)


def _result_word(code: int, left, right, result, raised: bool) -> int:
    if raised:
        actual = RAISED
    elif type(result) is bool:
        actual = int(result)
    else:
        return _ALL_UNKNOWN
    left_kind, right_kind = _kind(left), _kind(right)
    word = 0
    for other in range(10):
        if other == code:
            outcome = actual
        elif _NEGATION.get(code) == other and actual != RAISED:
            outcome = 1 - actual
        elif other in (6, 7):
            outcome = int((left is right) == (other == 6))
        elif other in (8, 9):
            membership = left_kind in (SCALAR, ENUM) and (right_kind == TUPLE or (right_kind == SCALAR and type(right) in (str, bytes)))
            outcome = UNKNOWN
            if membership:
                try:
                    outcome = int(evaluate(other, left, right))
                except Exception:
                    outcome = RAISED
        elif left_kind != OPAQUE and right_kind != OPAQUE:
            try:
                value = evaluate(other, left, right)
                outcome = int(value) if type(value) is bool else UNKNOWN
            except Exception:
                outcome = RAISED
        else:
            outcome = UNKNOWN
        word |= outcome << (2 * other)
    return word


class _Recorder:
    def __init__(self):
        self.on = False
        self.import_time: set[int] = set()
        self.words: dict[int, set[int]] = {}
        self.operands: dict[int, set | None] = {}
        self.memo: dict = {}

    def note(self, site: int, code: int, left, right, result, raised: bool) -> None:
        if not self.on:
            self.import_time.add(site)
            return
        try:
            key = None
            if (type(left) in _SCALARS or _type_kind(type(left)) == ENUM) and \
                    (type(right) in _SCALARS or _type_kind(type(right)) == ENUM) and not raised:
                key = (site, type(left), left, type(right), right, left is right)
            word = self.memo.get(key) if key is not None else None
            if word is None:
                word = _result_word(code, left, right, result, raised)
                if key is not None:
                    self.memo[key] = word
            self.words.setdefault(site, set()).add(word)
            side = SITES[site][6]
            if side:
                other = right if side == "L" else left
                kept = self.operands.get(site, ())
                if kept is not None:
                    if not _keepable(other):
                        self.operands[site] = None
                    else:
                        if kept == ():
                            kept = self.operands[site] = set()
                        if other not in kept:
                            if len(kept) >= 64:
                                self.operands[site] = None
                            else:
                                kept.add(other)
        except Exception:  # noqa
            self.words.setdefault(site, set()).add(_ALL_UNKNOWN)
            if SITES[site][6]:
                self.operands[site] = None


RECORDER = _Recorder()


def _compare(site, code, left, right):
    try:
        result = evaluate(code, left, right)
    except BaseException:
        RECORDER.note(site, code, left, right, None, True)
        raise
    RECORDER.note(site, code, left, right, result, False)
    return result


builtins.__forge_compare__ = _compare  # a dunder name: not mangled inside class bodies


def record(tests_json: str, out_dir: str, parallel: int = 4) -> None:
    common.enter_pinned_tree()
    sys.dont_write_bytecode = True
    os.makedirs(out_dir, exist_ok=True)
    tests = common.load_tests(tests_json)
    sys.meta_path.insert(0, _InstrumentingFinder())
    common.calm_the_process()
    common.import_everything(tests)
    assert SITES, "the instrumentation did not run"
    print(f"{len(SITES)} sites, {len(RECORDER.import_time)} evaluated while importing", file=sys.stderr, flush=True)
    with open(os.path.join(out_dir, "sites.json"), "w") as handle:
        json.dump({"sites": SITES, "import_time": sorted(RECORDER.import_time)}, handle)
    spawned = {"flag": False}
    sys.addaudithook(lambda event, args: spawned.__setitem__("flag", True)
                     if RECORDER.on and event in ("subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn") else None)

    def run(test):
        def work(output):
            known = len(SITES)
            RECORDER.on = True
            result = common.run_alone(test["id"])
            RECORDER.on = False
            row = {"number": test["number"], "id": test["id"], "ok": common.passed(result), "spawned": spawned["flag"],
                   "base": known, "new_sites": SITES[known:],
                   "words": {s: sorted(w) for s, w in RECORDER.words.items()},
                   "operands": {s: None if v is None else sorted(v, key=repr) for s, v in RECORDER.operands.items()}}
            common.write_atomically(output, lambda handle: marshal.dump(row, handle))
        return work

    common.fan_out([(os.path.join(out_dir, f"c-{t['number']}.bin"), run(t)) for t in common.heaviest_first(tests)],
                   parallel, alarm_s=5400)


def assemble(raw_dir: str, tests_json: str, output: str) -> None:
    from ripple.atlas import write_artifact

    tests = common.load_tests(tests_json)
    index = {t["number"]: i for i, t in enumerate(tests)}
    with open(os.path.join(raw_dir, "sites.json")) as handle:
        base = json.load(handle)
    known = [tuple(s) for s in base["sites"]]
    sites = {tuple(s[:5]): (s[5], s[6]) for s in known}
    import_time = {tuple(known[i][:5]) for i in base["import_time"]}
    results, operands, operands_lost = {}, {}, {}
    untrusted = recorded = 0
    for path in glob.glob(os.path.join(raw_dir, "c-*.bin")):
        with open(path, "rb") as handle:
            row = marshal.load(handle)
        if row["number"] not in index:
            continue
        test = 1 << index[row["number"]]
        recorded |= test
        if not row["ok"] or row["spawned"]:
            untrusted |= test
        extra = [tuple(s) for s in row["new_sites"]]
        for site in extra:
            sites.setdefault(site[:5], (site[5], site[6]))

        def key_of(number, _base=row["base"], _extra=extra):
            return known[number][:5] if number < _base else _extra[number - _base][:5]

        for number, words in row["words"].items():
            table = results.setdefault(key_of(number), {})
            for word in words:
                table[word] = table.get(word, 0) | test
        for number, values in row["operands"].items():
            key = key_of(number)
            if values is None:
                operands_lost[key] = operands_lost.get(key, 0) | test
            else:
                table = operands.setdefault(key, {})
                for value in values:
                    table[value] = table.get(value, 0) | test
    untrusted |= ((1 << len(tests)) - 1) & ~recorded  # no record at all
    write_artifact(output, {"test_ids": [t["id"] for t in tests], "sites": sites, "import_time": import_time,
                            "results": results, "operands": operands, "operands_lost": operands_lost,
                            "untrusted": untrusted, "files": sorted({k[0] for k in sites})})
    print(f"{len(sites)} sites ({len(import_time)} import-time); {untrusted.bit_count()} untrusted tests", file=sys.stderr)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "record":
        record(args[0], args[1], int(args[2]) if len(args) > 2 else 4)
    else:
        assemble(*args)
