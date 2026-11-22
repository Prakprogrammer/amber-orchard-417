"""The pass atlas: per-pass line coverage of tests made of independent passes.

    python -m forge.passes record /data/tests.json <raw.json> [parallel]
    python -m forge.passes pack <raw.json> atlas/passes.pkl.gz

Passes: the dialects of test_generator_memory, the queries of the TPC executor tests, and the
.sql fixture entries that fixture-driven tests (optimizer, transpile, serde) walk through the
loaders of tests.helpers (found by running each such test once with counting loaders). Each
pass then runs alone in a fresh fork with line coverage. A test with any failing pass is never
reduced (its entry is None).
"""

from __future__ import annotations

import importlib
import json
import os
import sys

from forge import common
from ripple.standins import FIXTURE_LOADERS, OnlyPasses, realize

FIXTURE_TEST_FILES = ("tests/test_optimizer.py", "tests/test_transpile.py", "tests/test_serde.py")
MIN_SECONDS = float(os.environ.get("FORGE_PASSES_MIN_SECONDS", "0.05"))
GENERATOR_MEMORY = "tests/test_generator.py::TestGenerator::test_generator_memory"
QUERY_LISTS = {"tests/test_executor.py::TestExecutor::test_execute_tpcds": "optimizer/tpc-ds/tpc-ds.sql",
               "tests/test_executor.py::TestExecutor::test_execute_tpch": "optimizer/tpc-h/tpc-h.sql",
               "tests/test_executor.py::TestExecutor::test_optimized_tpch": "optimizer/tpc-h/tpc-h.sql"}
SCRATCH = "/tmp/forge-passes"


def _looping_labels(test_id: str) -> list[tuple]:
    if test_id == GENERATOR_MEMORY:
        return [("dialect", d.value) for d in common.module_of(test_id).Dialects]
    from tests.helpers import load_sql_fixture_pairs

    return [("query", i) for i in range(len(list(load_sql_fixture_pairs(QUERY_LISTS[test_id]))))]


def _kind(label: tuple) -> str:
    return "fixture" if label[0] in FIXTURE_LOADERS else label[0]


def _discover(test_id: str):
    def work(output):
        module = common.module_of(test_id)
        counts: dict[tuple, int] = {}
        for kind, loader_name in FIXTURE_LOADERS.items():
            loader = getattr(module, loader_name, None)
            if loader is None:
                continue

            def counting(filename, _loader=loader, _kind=kind):
                n = 0
                for item in _loader(filename):
                    n += 1
                    yield item
                counts[(_kind, filename)] = max(counts.get((_kind, filename), 0), n)

            setattr(module, loader_name, counting)
        ok = common.passed(common.run_alone(test_id))
        with open(output, "w") as handle:
            json.dump({"ok": ok, "files": [[k, f, n] for (k, f), n in counts.items()]}, handle)
    return work


def _one_pass(test_id: str, label: tuple):
    def work(output):
        module = common.module_of(test_id)
        realize(test_id, OnlyPasses(frozenset([label]), _kind(label))).install(module)
        with common.recording_lines(prefix=common.PREFIX + "sqlglot/") as lines:
            ok = common.passed(common.run_alone(test_id))
        with open(output, "w") as handle:
            json.dump({"ok": ok, "lines": common.by_file(lines)}, handle)
    return work


def record(tests_json: str, output: str, parallel: int = 4) -> None:
    common.enter_pinned_tree()
    tests = common.load_tests(tests_json)
    common.calm_the_process()
    import sqlglot  # noqa: F401

    looping = [GENERATOR_MEMORY, *QUERY_LISTS]
    for name in sorted({t.split("::")[0] for t in looping} | set(FIXTURE_TEST_FILES)):
        importlib.import_module(name[:-3].replace("/", "."))
    os.makedirs(SCRATCH, exist_ok=True)
    fixture_tests = [t["id"] for t in tests if t["file"] in FIXTURE_TEST_FILES and t["seconds"] >= MIN_SECONDS
                     and t["id"] not in looping]
    common.fan_out([(os.path.join(SCRATCH, f"d{i}.json"), _discover(t)) for i, t in enumerate(fixture_tests)],
                   parallel, alarm_s=1800, report_every=0)
    work = [(t, label) for t in looping for label in _looping_labels(t)]
    for i, test_id in enumerate(fixture_tests):
        path = os.path.join(SCRATCH, f"d{i}.json")
        if os.path.exists(path):
            with open(path) as handle:
                found = json.load(handle)
            if found["ok"]:
                work += [(test_id, (kind, fixture, k)) for kind, fixture, n in found["files"] for k in range(n)]
    print(f"{len(fixture_tests)} fixture tests; {len(work)} passes", file=sys.stderr, flush=True)
    common.fan_out([(os.path.join(SCRATCH, f"p{i}.json"), _one_pass(t, label)) for i, (t, label) in enumerate(work)],
                   parallel, alarm_s=1800)
    raw: dict[str, list] = {}
    for i, (test_id, label) in enumerate(work):
        path = os.path.join(SCRATCH, f"p{i}.json")
        row = json.load(open(path)) if os.path.exists(path) else None
        raw.setdefault(test_id, []).append([list(label), row["lines"] if row and row["ok"] else None])
    with open(output, "w") as handle:
        json.dump(raw, handle)


def pack(raw_path: str, output: str) -> None:
    """{test: [(label, lines or None)]} -> {test: {"labels", "reach": {path: {line: pass bits}}}}."""
    from ripple.atlas import write_artifact

    with open(raw_path) as handle:
        raw = json.load(handle)
    packed = {}
    for test_id, rows in raw.items():
        rows = sorted(((tuple(label), lines) for label, lines in rows), key=lambda r: r[0])
        if any(lines is None for _, lines in rows):
            packed[test_id] = None
            continue
        reach: dict[str, dict[int, int]] = {}
        for bit, (_, lines) in enumerate(rows):
            for path, numbers in lines.items():
                hits = reach.setdefault(path, {})
                for number in numbers:
                    hits[number] = hits.get(number, 0) | (1 << bit)
        packed[test_id] = {"labels": [label for label, _ in rows], "reach": reach}
    write_artifact(output, packed)
    print(f"{len(packed)} tests", file=sys.stderr)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "record":
        record(args[0], args[1], int(args[2]) if len(args) > 2 else 4)
    else:
        pack(*args)
