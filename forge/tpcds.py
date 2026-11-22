"""The TPC-DS record for ripple.tpcds, from the pinned tree.

    python -m forge.tpcds record <out.pkl.gz> [parallel]
    python -m forge.tpcds merge <a.pkl.gz> <b.pkl.gz> atlas/tpcds.pkl.gz

``record`` runs every query of tests/fixtures/optimizer/tpc-ds/tpc-ds.sql marked ``execute`` in a
fresh fork and keeps the digests of its DuckDB text, optimized expression and rows, plus the
sqlglot lines Plan() and PythonExecutor.execute() ran. Record twice under different
PYTHONHASHSEED values and ``merge``: a query whose digests differ is marked unstable.
"""

from __future__ import annotations

import json
import os
import sys

from forge import common
from ripple import tpcds as stage
from ripple.atlas import read_artifact, write_artifact

SCRATCH = "/tmp/forge-tpcds"


def record(output: str, parallel: int = 4) -> None:
    common.enter_pinned_tree()
    common.calm_the_process()
    import tests.test_executor as module
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.executor.table import ensure_tables
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema

    klass = module.TestExecutor
    klass.setUpClass()
    queries = klass.tpcds_sqls
    os.makedirs(SCRATCH, exist_ok=True)

    def one(index):
        def work(path):
            meta, sql, _ = queries[index]
            row = {"text": stage.digest(sql), "execute": bool(meta.get("execute"))}
            if row["execute"]:
                parsed = parse_one(sql)
                row["duckdb_text"] = stage.digest(transpile(sql, write="duckdb")[0])
                tables = ensure_tables({t.name: klass.tpcds_tables[t.name] for t in find_tables(parsed)}, dialect=None)
                optimized = optimize(parsed, ensure_schema(module.TPCDS_SCHEMA, dialect=None),
                                     leave_tables_isolated=True, dialect=None)
                row["optimized"] = stage.expression_digest(optimized)
                with common.recording_lines(prefix=common.PREFIX + "sqlglot/") as lines:
                    result = PythonExecutor(tables=tables).execute(Plan(optimized))
                row["rows"] = stage.rows_digest(result)
                row["executor_lines"] = common.by_file(lines)
            with open(path, "w") as handle:
                json.dump(row, handle)
        return work

    common.fan_out([(os.path.join(SCRATCH, f"q{i}.json"), one(i)) for i in range(len(queries))], parallel,
                   alarm_s=900, report_every=0)
    rows = []
    for i in range(len(queries)):
        path = os.path.join(SCRATCH, f"q{i}.json")
        if not os.path.exists(path):
            rows.append(None)
            continue
        with open(path) as handle:
            row = json.load(handle)
        if row.get("executor_lines") is not None:
            row["executor_lines"] = {p: set(ls) for p, ls in row["executor_lines"].items()}
        row["stable"] = True
        rows.append(row)
    write_artifact(output, rows)
    print(f"{sum(r is not None for r in rows)} of {len(queries)} queries recorded", file=sys.stderr)


def merge(first: str, second: str, output: str) -> None:
    a, b = read_artifact(first), read_artifact(second)
    unstable = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if x.get("execute") and any(x.get(k) != y.get(k) for k in ("duckdb_text", "optimized", "rows")):
            x["stable"] = False
            unstable += 1
        if x.get("executor_lines") is not None and y.get("executor_lines") is not None:
            for path, lines in y["executor_lines"].items():
                x["executor_lines"].setdefault(path, set()).update(lines)
    write_artifact(output, a)
    print(f"merged; {unstable} unstable queries", file=sys.stderr)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "record":
        record(args[0], int(args[1]) if len(args) > 1 else 4)
    else:
        merge(*args)
