"""Cheaper runs that decide a slow test's verdict for it.

The planner attaches a *directive* (a small frozen record) to a test; inside the child,
``realize`` turns it into a *stand-in* that edits the test module while the test runs:

- ``OnlyPasses``: run only some independent passes of a test (dialects of a loop, queries of a
  TPC list, entries of .sql fixtures). A pass that runs no edited line behaves as on the pinned
  commit, where every pass passes, so the test fails iff one of the kept passes fails.
- ``Miniature``: tests that only check behaviour at a size run a copy of their method, compiled
  from the pinned source with smaller constants.
- ``TpcdsStageCheck``: test_execute_tpcds is replaced by the stage check of ``ripple.tpcds``.
"""

from __future__ import annotations

from dataclasses import dataclass

TPCDS_TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
FIXTURE_LOADERS = {"pairs": "load_sql_fixture_pairs", "lines": "load_sql_fixtures"}


# ---- directives (made by the planner) ----------------------------------------------------
@dataclass(frozen=True)
class OnlyPasses:
    labels: frozenset  # pass labels: ("dialect", name) | ("query", i) | ("pairs"|"lines", fixture, i)
    kind: str  # "dialect" | "query" | "fixture"


@dataclass(frozen=True)
class Miniature:
    pass


@dataclass(frozen=True)
class TpcdsStageCheck:
    edited_lines: dict | None  # {path: lines} of the patch; None: unknown, check every query
    queries: frozenset | None  # query indices whose recorded lines reach an edit; None: all


# ---- stand-ins (live in the child) ----------------------------------------------------------
class _StandIn:
    around_each_test = True

    def install(self, module) -> None:
        raise NotImplementedError

    def uninstall(self, module) -> None:
        pass


def _pinned_attribute(holder, name: str):
    """The attribute as first seen (kept aside under a private name, so stand-ins can stack)."""
    key = "_ripple_pinned_" + name
    if not hasattr(holder, key):
        setattr(holder, key, getattr(holder, name))
    return getattr(holder, key)


class _DialectSubset(_StandIn):
    """test_generator_memory loops ``for name in Dialects``: keep only some dialects."""

    def __init__(self, names):
        self.names = {str(name) for name in names}

    def install(self, module) -> None:
        module.Dialects = [d for d in _pinned_attribute(module, "Dialects") if d.value in self.names]

    def uninstall(self, module) -> None:
        if hasattr(module, "_ripple_pinned_Dialects"):
            module.Dialects = module._ripple_pinned_Dialects


class _QuerySubset(_StandIn):
    """Executor tests iterating a query list that setUpClass fills (``cls.tpcds_sqls``...).
    Installed once for the class: the filter wraps its setUpClass, and tests sharing a list
    share the union of their kept queries."""

    around_each_test = False
    LISTS = {
        "tests/test_executor.py::TestExecutor::test_execute_tpcds": ("TestExecutor", "tpcds_sqls"),
        "tests/test_executor.py::TestExecutor::test_execute_tpch": ("TestExecutor", "tpch_sqls"),
        "tests/test_executor.py::TestExecutor::test_optimized_tpch": ("TestExecutor", "tpch_sqls"),
    }

    def __init__(self, test_id: str, indices):
        self.class_name, self.attribute = self.LISTS[test_id]
        self.indices = {int(i) for i in indices}

    def install(self, module) -> None:
        klass = getattr(module, self.class_name)
        filters = klass.__dict__.get("_ripple_query_filters")
        if filters is None:
            filters = {}
            pinned_setup = klass.__dict__["setUpClass"].__func__

            def setUpClass(cls, _pinned=pinned_setup, _filters=filters):
                _pinned(cls)
                for attribute, kept in _filters.items():
                    items = getattr(cls, attribute)
                    setattr(cls, attribute, [items[i] for i in kept if i < len(items)])

            klass._ripple_query_filters = filters
            klass.setUpClass = classmethod(setUpClass)
        filters[self.attribute] = sorted(self.indices | set(filters.get(self.attribute, ())))


class _FixtureEntrySubset(_StandIn):
    """Fixture-driven tests read .sql fixtures through the loaders of tests.helpers; wrap the
    loaders in the test module's namespace so only kept entries come out."""

    def __init__(self, labels):
        self.labels = set(labels)

    def install(self, module) -> None:
        for kind, loader_name in FIXTURE_LOADERS.items():
            pinned = getattr(module, "_ripple_pinned_" + loader_name, None) or getattr(module, loader_name, None)
            if pinned is None:
                continue
            setattr(module, "_ripple_pinned_" + loader_name, pinned)

            def kept_entries(filename, _pinned=pinned, _kind=kind):
                for index, item in enumerate(_pinned(filename)):
                    if (_kind, filename, index) in self.labels:
                        yield item

            setattr(module, loader_name, kept_entries)

    def uninstall(self, module) -> None:
        for loader_name in FIXTURE_LOADERS.values():
            pinned = getattr(module, "_ripple_pinned_" + loader_name, None)
            if pinned is not None:
                setattr(module, loader_name, pinned)


class _Miniature(_StandIn):
    """The same method with smaller constants (the replaced text must match exactly once)."""

    SHRINK = {
        "tests/test_parser.py::TestParser::test_max_nodes": [
            ("width = 10000", "width = 20"),
            # 100 * width + 1 counted nodes: one more than the first limit, as in the real test
            ("            parse_one(sql, max_nodes=1_000_000)", "            parse_one(sql, max_nodes=2_000)"),
            ("parse_one(sql, max_nodes=50_000)", "parse_one(sql, max_nodes=1_000)"),
            ('"(50000)"', '"(1000)"'),
        ],
        "tests/test_generator.py::TestGenerator::test_generator_memory": [
            # the warm-up stays whole: lazy per-dialect state must exist before measuring
            ("for _ in range(1000):", "for _ in range(60):"),
        ],
        "tests/test_optimizer.py::TestOptimizer::test_scope": [
            ('["SELECT x FROM t"] * 10000))', '["SELECT x FROM t"] * 100))'),
            ('" UNION SELECT 1 INTERSECT SELECT 1" * 1000, read="hive")',
             '" UNION SELECT 1 INTERSECT SELECT 1" * 20, read="hive")'),
            ("traverse())), 4001)", "traverse())), 81)"),
        ],
    }

    def __init__(self, test_id: str):
        self.test_id = test_id
        self.replaced = None

    def install(self, module) -> None:
        import inspect
        import textwrap

        _, class_name, method_name = self.test_id.split("::")
        klass = getattr(module, class_name)
        pinned = klass.__dict__[method_name]
        lines, first_line = inspect.getsourcelines(pinned)
        source = "".join(lines)
        for old, new in self.SHRINK[self.test_id]:
            if source.count(old) != 1:
                raise ValueError(f"{self.test_id}: not the pinned text ({old!r})")
            source = source.replace(old, new)
        namespace: dict = {}
        code = compile("\n" * (first_line - 1) + textwrap.dedent(source), inspect.getsourcefile(pinned), "exec")
        exec(code, module.__dict__, namespace)
        self.replaced = (klass, method_name, pinned)
        setattr(klass, method_name, namespace[method_name])

    def uninstall(self, module) -> None:
        if self.replaced is not None:
            klass, method_name, pinned = self.replaced
            setattr(klass, method_name, pinned)
            self.replaced = None


class _TpcdsStageCheck(_StandIn):
    """While it runs, test_execute_tpcds is the stage check: a pass means every query would
    give the pinned rows; no verdict fails the stand-in, so the real test gets listed."""

    def __init__(self, directive: TpcdsStageCheck):
        self.directive = directive
        self.replaced = None

    def install(self, module) -> None:
        from . import tpcds
        from .harness import VERDICTS

        record = tpcds.load_record()
        klass = module.TestExecutor
        directive = self.directive

        def test_execute_tpcds(case):
            if VERDICTS.nothing_points_at(TPCDS_TEST):
                return  # no failure in its class and few overall: not listed, at no cost
            if record is None:
                raise tpcds.NoVerdict("no stage record")
            tpcds.check_stages(module, type(case), record, directive.edited_lines, directive.queries, case)

        self.replaced = (klass, klass.__dict__["test_execute_tpcds"])
        klass.test_execute_tpcds = test_execute_tpcds

    def uninstall(self, module) -> None:
        if self.replaced is not None:
            klass, pinned = self.replaced
            klass.test_execute_tpcds = pinned
            self.replaced = None


def realize(test_id: str, directive) -> _StandIn:
    if isinstance(directive, Miniature):
        return _Miniature(test_id)
    if isinstance(directive, TpcdsStageCheck):
        return _TpcdsStageCheck(directive)
    if directive.kind == "dialect":
        return _DialectSubset(label[1] for label in directive.labels)
    if directive.kind == "query":
        return _QuerySubset(test_id, (label[1] for label in directive.labels))
    return _FixtureEntrySubset(directive.labels)


def directive_for_passes(labels, kind: str) -> OnlyPasses:
    return OnlyPasses(frozenset(labels), kind)


def pass_kind(labels) -> str:
    """The kind of a test's passes, from its first label."""
    first = next(iter(labels), None)
    if first is None or first[0] in FIXTURE_LOADERS:
        return "fixture"
    return first[0]
