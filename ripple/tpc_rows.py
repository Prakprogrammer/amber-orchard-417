"""Stored TPC rows for the class setup of tests/test_executor.py::TestExecutor.

That setUpClass reads every TPC-H and TPC-DS csv file into Python rows (about 6 s of CPU)
before any executor test runs. The rows depend only on the csv files of the pinned tree, so the
image build stores them once, after checking that a setup made from the stored rows gives
exactly what the real setup gives:

    python -m ripple.tpc_rows <output path>

In a child running executor tests, the class setup is replaced by one that loads those rows
and otherwise does what the real one does: the same DuckDB views (made on first use), Table
objects made by the live, possibly patched, Table class, the same fixture lists. Never used
when a patch touches the test file, the test helpers or the TPC fixtures.
"""

from __future__ import annotations

import os
import pickle
import sys

_DEPENDS_ON = ("tests/test_executor.py", "tests/helpers.py", "tests/fixtures/optimizer/tpc-h/",
               "tests/fixtures/optimizer/tpc-ds/")
_BENCHMARKS = (("tpch", "DIR_TPCH", "TPCH_SCHEMA"), ("tpcds", "DIR_TPCDS", "TPCDS_SCHEMA"))
_rows_by_path: dict[str, dict] = {}


def usable(path: str, new_contents: dict) -> bool:
    return os.path.exists(path) and not any(p.startswith(_DEPENDS_ON) for p in new_contents)


class _ConnectionOnDemand:
    """A DuckDB connection whose views over the csv files are made on first use (most
    executor tests never query DuckDB, and making the views opens every csv file)."""

    def __init__(self, connect):
        self._connect = connect
        self._connection = None

    def close(self):
        if self._connection is not None:
            self._connection.close()

    def __getattr__(self, name):
        if self._connection is None:
            self._connection = self._connect()
        return getattr(self._connection, name)


def _views_over(module, directory: str, schema: dict):
    def connect():
        connection = module.duckdb.connect()
        for table, columns in schema.items():
            file_name = f"{directory}{table}.csv.gz"
            connection.execute(
                f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})
                """
            )
        return connection

    return connect


def install_cached_setup(module, path: str) -> None:
    """Replace TestExecutor.setUpClass of the imported tests.test_executor module."""
    klass = getattr(module, "TestExecutor", None)
    if klass is None or "setUpClass" not in klass.__dict__:
        return

    def setUpClass(cls):
        if path not in _rows_by_path:
            with open(path, "rb") as handle:
                _rows_by_path[path] = pickle.load(handle)
        stored = _rows_by_path[path]
        for key, directory_name, schema_name in _BENCHMARKS:
            directory, schema = getattr(module, directory_name), getattr(module, schema_name)
            setattr(cls, f"{key}_conn", _ConnectionOnDemand(_views_over(module, directory, schema)))
        for key, directory_name, schema_name in _BENCHMARKS:
            schema = getattr(module, schema_name)
            tables = {table: module.Table(columns=columns, rows=list(stored[key][table])) for table, columns in schema.items()}
            setattr(cls, f"{key}_tables", tables)
        cls.cache = {}
        cls.tpch_sqls = list(module.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
        cls.tpcds_sqls = list(module.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))

    klass.setUpClass = classmethod(setUpClass)


def _what_setup_made(klass):
    tables = {key: {name: (tuple(t.columns), list(t.rows)) for name, t in getattr(klass, f"{key}_tables").items()}
              for key, _, _ in _BENCHMARKS}
    return tables, (klass.tpch_sqls, klass.tpcds_sqls)


def build(output: str, sqlglot_root: str) -> None:
    import importlib

    sys.path.insert(0, sqlglot_root)
    os.chdir(sqlglot_root)
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    module = importlib.import_module("tests.test_executor")
    klass = module.TestExecutor
    klass.setUpClass()
    real_tables, real_lists = _what_setup_made(klass)
    klass.tearDownClass()
    rows = {key: {name: table_rows for name, (_, table_rows) in tables.items()} for key, tables in real_tables.items()}
    partial = output + ".partial"
    with open(partial, "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # the proof: a setup from the stored rows must give exactly the real tables and lists
    install_cached_setup(module, partial)
    klass.setUpClass()
    cached_tables, cached_lists = _what_setup_made(klass)
    for key, _, _ in _BENCHMARKS:
        connection = getattr(klass, f"{key}_conn")
        for table in list(getattr(klass, f"{key}_tables"))[:3]:
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchall()
    klass.tearDownClass()
    if (cached_tables, cached_lists) != (real_tables, real_lists):
        os.remove(partial)
        raise SystemExit("stored TPC rows do not reproduce the real setup: not used")
    os.replace(partial, output)
    print(f"tpc rows: {sum(len(v) for v in rows.values())} tables, {os.path.getsize(output) / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    build(sys.argv[1], os.environ.get("RIPPLE_SQLGLOT", "/opt/sqlglot"))
