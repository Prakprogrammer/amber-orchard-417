"""The child side of a run: execute chosen unittest methods and report each verdict.

One child runs one commit's tests, after either swapping patched function bodies into the
preloaded modules (``Swap``) or switching to a mirrored, patched tree. It writes one JSON
report per line to a pipe:

    {"kind": "verdict", "test": id, "verdict": "pass"|"fail"|"timeout", "wall": s, "cpu": s}
    {"kind": "ready"} | {"kind": "class_setup"|"module_import", "key": name, "cpu": s}
    {"kind": "class_error", "key": "file::Class", "note": text}

Every test has a CPU limit. Passing it raises ``CpuLimit`` inside the test; the test is then
reported as a timeout, which lists it.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field

from .atlas import test_class, test_module

_STOP_ON_SIGNAL = {"fired": False}


class CpuLimit(KeyboardInterrupt):
    """Raised by the CPU guard. A KeyboardInterrupt so that unittest's subTest and error
    handling let it pass (they swallow ordinary exceptions)."""


@dataclass
class VerdictLog:
    """What this child has seen so far (stand-ins decide from it)."""

    started: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    many_failures: int = int(os.environ.get("RIPPLE_MANY_FAILURES", "100000"))

    def nothing_points_at(self, test_id: str) -> bool:
        """The evidence rule for a held test: no failure in its class and fewer than
        ``many_failures`` failures overall; with no sibling run at all, no failure anywhere."""
        klass = test_class(test_id) + "::"
        others_failed = [t for t in self.failed if t != test_id]
        if any(t.startswith(klass) for t in others_failed):
            return False
        sibling_ran = any(t.startswith(klass) and t != test_id for t in self.started)
        if not self.started or (not sibling_ran and others_failed):
            return False
        return len(others_failed) < self.many_failures


VERDICTS = VerdictLog()


def run_pools_inline() -> None:
    """Tests hand work to spawned process pools, whose start-up costs seconds of CPU. Run the
    submitted calls here instead, pickling calls and results the way a real pool would, so that
    sqlglot's serialization is exercised exactly as in a real run."""
    import concurrent.futures as futures
    import concurrent.futures.process as process_futures
    import pickle

    class InlinePool(futures.Executor):
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
            if initializer is not None:
                initializer(*initargs)

        def submit(self, fn, /, *args, **kwargs):
            future = futures.Future()
            try:
                fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
                future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
            except CpuLimit:
                raise
            except BaseException as problem:  # noqa
                future.set_exception(problem)
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            pass

    futures.ProcessPoolExecutor = InlinePool
    process_futures.ProcessPoolExecutor = InlinePool


class _Reporter:
    def __init__(self, fd: int):
        self.stream = os.fdopen(fd, "w", buffering=1)

    def send(self, kind: str, **fields) -> None:
        self.stream.write(json.dumps({"kind": kind, **fields}) + "\n")

    def verdict(self, test_id: str, verdict: str, wall: float = 0.0, cpu: float = 0.0, note: str = "") -> None:
        self.send("verdict", test=test_id, verdict=verdict, wall=round(wall, 3), cpu=round(cpu, 3), note=note[:200])

    def close(self) -> None:
        self.stream.close()


@dataclass
class ChildOrders:
    tree: str  # the mirrored tree (unused when swapping)
    pinned_root: str
    test_ids: list[str]
    cpu_limits: dict[str, float]
    reimport_sqlglot: bool
    swap: object = None  # hotswap.Swap or None
    stand_ins: dict = field(default_factory=dict)  # test id -> stand-in directive
    use_tpc_rows: bool = False
    tpc_rows_path: str = ""


def _arm(cpu_limit: float) -> None:
    signal.setitimer(signal.ITIMER_PROF, cpu_limit)
    signal.setitimer(signal.ITIMER_REAL, 30.0 + 4.0 * cpu_limit)


def _disarm() -> None:
    signal.setitimer(signal.ITIMER_PROF, 0)
    signal.setitimer(signal.ITIMER_REAL, 0)


def _on_alarm(signum, frame):
    _STOP_ON_SIGNAL["fired"] = True
    raise CpuLimit()


def child_main(orders: ChildOrders, fd: int) -> None:
    import importlib

    from .sandbox import enter_tree
    from .standins import realize

    reporter = _Reporter(fd)
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)  # tests must not write to our stdout
    if orders.swap is not None:
        os.chdir(orders.pinned_root)
        orders.swap.apply()
    else:
        enter_tree(orders.tree, orders.pinned_root)
        for name in list(sys.modules):
            if name == "tests" or name.startswith("tests."):
                del sys.modules[name]
            elif orders.reimport_sqlglot and (name == "sqlglot" or name.startswith("sqlglot.")):
                del sys.modules[name]
        importlib.invalidate_caches()
    os.cpu_count = lambda: 2
    if hasattr(os, "process_cpu_count"):
        os.process_cpu_count = lambda: 2
    run_pools_inline()
    if orders.reimport_sqlglot and orders.swap is None:
        import sqlglot  # noqa: F401  (the patched tree's import, timed into readiness)
    reporter.send("ready")
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.signal(signal.SIGPROF, _on_alarm)

    by_class: dict[tuple[str, str], list[str]] = {}
    for test_id in orders.test_ids:
        file_name, class_name, _ = test_id.split("::")
        by_class.setdefault((file_name, class_name), []).append(test_id)
    around_each: dict[str, tuple] = {}  # test id -> (stand-in, module), installed while it runs
    for (file_name, class_name), test_ids in by_class.items():
        _run_class(orders, reporter, file_name, class_name, test_ids, around_each, realize)
    reporter.close()


def _run_class(orders, reporter, file_name, class_name, test_ids, around_each, realize) -> None:
    import importlib
    import unittest

    module_name = test_module(test_ids[0])
    try:
        signal.setitimer(signal.ITIMER_REAL, 60.0)
        first_import = module_name not in sys.modules
        cpu_before = time.process_time()
        module = importlib.import_module(module_name)
        signal.setitimer(signal.ITIMER_REAL, 0)
        if first_import:
            reporter.send("module_import", key=module_name, cpu=round(time.process_time() - cpu_before, 3))
        klass = getattr(module, class_name)
        if orders.use_tpc_rows and module_name == "tests.test_executor":
            from .tpc_rows import install_cached_setup

            install_cached_setup(module, orders.tpc_rows_path)
        for test_id in test_ids:
            directive = orders.stand_ins.get(test_id)
            if directive is None:
                continue
            stand_in = realize(test_id, directive)
            if stand_in.around_each_test:
                around_each[test_id] = (stand_in, module)  # module-wide edits: only while it runs
            else:
                stand_in.install(module)
    except BaseException as problem:  # an import failure fails every test of the file
        signal.setitimer(signal.ITIMER_REAL, 0)
        for test_id in test_ids:
            reporter.verdict(test_id, "fail", note=f"import: {type(problem).__name__}: {problem}")
        return
    test_of: dict[int, str] = {}
    cases = []
    for test_id in test_ids:
        try:
            case = klass(test_id.split("::")[2])
        except BaseException as problem:
            reporter.verdict(test_id, "fail", note=f"init: {problem}")
            continue
        test_of[id(case)] = test_id
        cases.append(case)
    if not cases:
        return
    result = _Result(test_of, orders.cpu_limits, reporter, around_each, f"{file_name}::{class_name}")
    remaining = cases
    while remaining:
        try:
            result._testRunEntered = False
            unittest.TestSuite(remaining).run(result)  # a top-level run also tears the class down
            remaining = []
        except CpuLimit:
            _disarm()  # the guard stopped a test (reported as a timeout): go on with the rest
            unstarted = [case for case in remaining if test_of[id(case)] not in result.started]
            if len(unstarted) == len(remaining):  # stopped outside any test: in the class setup
                for case in unstarted:
                    reporter.verdict(test_of[id(case)], "fail", note="suite: stopped in class setup")
                unstarted = []
            _STOP_ON_SIGNAL["fired"] = False
            remaining = unstarted
        except BaseException as problem:
            _disarm()
            for test_id in test_ids:
                if test_id not in result.started:
                    reporter.verdict(test_id, "fail", note=f"suite: {problem}")
            remaining = []
    _disarm()
    if result.class_errors:
        note = "; ".join(result.class_errors)
        for test_id in test_ids:
            if test_id not in result.started or test_id not in result.failed:
                reporter.verdict(test_id, "fail", note="class fixture: " + note)
        reporter.send("class_error", key=f"{file_name}::{class_name}", note=note)


def _make_result_class():
    import unittest

    class Result(unittest.TestResult):
        def __init__(self, test_of, cpu_limits, reporter, around_each, class_key):
            super().__init__()
            self.test_of = test_of
            self.cpu_limits = cpu_limits
            self.reporter = reporter
            self.around_each = around_each
            self.class_key = class_key
            self.failed: set[str] = set()
            self.timed_out: set[str] = set()
            self.started: dict[str, float] = {}
            self.class_errors: list[str] = []
            self.setup_cpu_from: float | None = time.process_time()
            self.cpu_from = 0.0

        def _uninstall_all(self) -> None:
            for stand_in, module in self.around_each.values():
                stand_in.uninstall(module)

        def startTest(self, test):
            test_id = self.test_of.get(id(test))
            if test_id is not None:
                VERDICTS.started.append(test_id)
            self._uninstall_all()
            if test_id in self.around_each:
                stand_in, module = self.around_each[test_id]
                stand_in.install(module)
            if self.setup_cpu_from is not None:  # the class setup ran just before the first test
                self.reporter.send("class_setup", key=self.class_key, cpu=round(time.process_time() - self.setup_cpu_from, 3))
                self.setup_cpu_from = None
            super().startTest(test)
            self.started[test_id] = time.perf_counter()
            self.cpu_from = time.process_time()
            _arm(self.cpu_limits.get(test_id, 10.0))

        def stopTest(self, test):
            _disarm()
            self._uninstall_all()
            super().stopTest(test)
            test_id = self.test_of.get(id(test))
            wall = time.perf_counter() - self.started.get(test_id, time.perf_counter())
            cpu = time.process_time() - self.cpu_from
            if _STOP_ON_SIGNAL["fired"]:
                _STOP_ON_SIGNAL["fired"] = False
                self.timed_out.add(test_id)
                self.failed.add(test_id)
            verdict = "timeout" if test_id in self.timed_out else "fail" if test_id in self.failed else "pass"
            self.reporter.verdict(test_id, verdict, wall, cpu)

        def _failure(self, test, err) -> None:
            test_id = self.test_of.get(id(test))
            if test_id is None:  # setUpClass / tearDownClass / module fixtures
                self.class_errors.append(str(test))
                return
            VERDICTS.failed.append(test_id)
            self.failed.add(test_id)
            if err is not None and err[0] is CpuLimit:
                self.timed_out.add(test_id)

        def addError(self, test, err):
            self._failure(test, err)

        def addFailure(self, test, err):
            self._failure(test, err)

        def addSubTest(self, test, subtest, err):
            if err is not None:
                self._failure(test, err)

        def addUnexpectedSuccess(self, test):
            self._failure(test, None)

    return Result


_ResultClass = None


def _Result(*args):
    global _ResultClass
    if _ResultClass is None:
        _ResultClass = _make_result_class()
    return _ResultClass(*args)
