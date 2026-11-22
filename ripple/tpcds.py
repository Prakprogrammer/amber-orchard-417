"""The stage check that stands in for tests/test_executor.py::TestExecutor::test_execute_tpcds.

The real test parses all 99 TPC-DS queries and, for the 57 marked ``execute``, runs
sqlglot.executor.execute() on the TPC-DS tables (optimize -> Plan -> PythonExecutor) and
compares the rows with DuckDB's answer to the DuckDB translation of the query. Nearly all of
its time is the executor's row work.

The record (``forge.tpcds``, from the pinned tree) keeps, for every executed query, digests
of the DuckDB text, of the optimized expression and of the result rows, plus the sqlglot lines
that the plan and executor stages ran. The check repeats the steps with the patched code:
- the DuckDB text differs: DuckDB's answer is unknown -> no verdict (unless the test case is
  at hand, which then compares with DuckDB itself);
- the optimized expression is the pinned one and the patch edits none of the query's
  executor-stage lines: the executor gets the same input and runs the same code -> pinned rows;
- otherwise the query is executed here and its rows must equal the pinned rows.
Equal rows and the same DuckDB text feed the final comparison the same two inputs as on the
pinned commit, where it passes. Any exception fails the test, as it would in a real run.
"""

from __future__ import annotations

import hashlib
import json
import os

from .atlas import read_artifact

_DEFAULT_ATLAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atlas")
RECORD_PATH = os.path.join(os.environ.get("RIPPLE_ATLAS", _DEFAULT_ATLAS), "tpcds.pkl.gz")


class NoVerdict(Exception):
    pass


# The digests are stored in the record: their recipes must stay exactly as they are.
def digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


def expression_digest(expression) -> str:
    return digest(json.dumps(expression.dump(), sort_keys=True, default=repr) + "|" + expression.sql())


def rows_digest(table) -> str:
    return digest(repr((tuple(table.columns), [tuple(row) for row in table.rows])))


def load_record():
    return read_artifact(RECORD_PATH) if os.path.exists(RECORD_PATH) else None


def check_stages(module, klass, record: list, edited_lines: dict | None, queries: frozenset | None, case=None) -> None:
    """Raise NoVerdict when undecided; return when every query would give the pinned rows.
    ``queries``: the indices whose recorded lines reach an edit (None: all); the rest run as
    on the pinned commit. ``case``: when given, a query the shortcuts can't settle gets the
    test's own comparison with DuckDB."""
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.executor.table import ensure_tables
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema

    fixtures = klass.tpcds_sqls
    if len(fixtures) != len(record):
        raise NoVerdict("the fixture list changed")
    for index, (meta, sql, _) in enumerate(fixtures):
        pinned = record[index]
        if pinned is None or pinned["text"] != digest(sql):
            raise NoVerdict(f"query {index} has no record")
        if queries is not None and index not in queries:
            continue
        parsed = parse_one(sql)  # as the test does for every query (an error fails it)
        if not meta.get("execute"):
            continue
        if not pinned["stable"]:
            raise NoVerdict(f"query {index} is not reproducible")
        same_duckdb_text = digest(transpile(sql, write="duckdb")[0]) == pinned["duckdb_text"]
        tables = ensure_tables({t.name: klass.tpcds_tables[t.name] for t in find_tables(parsed)}, dialect=None)
        schema = ensure_schema(module.TPCDS_SCHEMA, dialect=None)
        optimized = optimize(parsed, schema, leave_tables_isolated=True, dialect=None)
        same_plan_input = expression_digest(optimized) == pinned["optimized"]
        stages_edited = edited_lines is None or any(
            not lines.isdisjoint(pinned["executor_lines"].get(path, ())) for path, lines in edited_lines.items())
        if same_plan_input and not stages_edited and same_duckdb_text:
            continue
        result = PythonExecutor(tables=tables).execute(Plan(optimized))
        if same_duckdb_text and rows_digest(result) == pinned["rows"]:
            continue
        if case is None:
            raise NoVerdict(f"query {index}: other rows or another DuckDB text")
        case.subtestHelper(index, result, tpch=False)
