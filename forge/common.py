"""What the atlas builders share: the pinned tree, forked one-test-per-process runs, recorders.

Builders run inside the forge image (``forge/Dockerfile``) with the pinned sqlglot checkout at
``$RIPPLE_SQLGLOT`` (default /opt/sqlglot) and this repository on ``sys.path``.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import time
from contextlib import contextmanager

PINNED_ROOT = os.environ.get("RIPPLE_SQLGLOT", "/opt/sqlglot")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = PINNED_ROOT.rstrip("/") + "/"


def enter_pinned_tree() -> None:
    for path in (PINNED_ROOT, REPO):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(PINNED_ROOT)
    os.environ["SKIP_INTEGRATION"] = "0"


def load_tests(path: str) -> list[dict]:
    with open(path) as handle:
        return json.load(handle)["tests"]


def module_of(test_id: str):
    return sys.modules[test_id.split("::")[0][:-3].replace("/", ".")]


def calm_the_process() -> None:
    """Two CPUs reported, process pools run inline: as in ripple's own children."""
    from ripple.harness import run_pools_inline

    os.cpu_count = lambda: 2
    os.process_cpu_count = lambda: 2
    run_pools_inline()


def import_everything(tests: list[dict] | None = None, strict_tests: bool = True) -> None:
    import pkgutil

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        try:
            importlib.import_module(info.name)
        except Exception as problem:  # noqa
            print(f"skipped {info.name}: {problem!r}", file=sys.stderr)
    for name in sorted({t["file"][:-3].replace("/", ".") for t in tests or ()}):
        try:
            importlib.import_module(name)
        except Exception:
            if strict_tests:
                raise


def run_alone(test_id: str):
    """Run one unittest method; returns the TestResult."""
    import unittest

    _, class_name, method = test_id.split("::")
    result = unittest.TestResult()
    unittest.TestSuite([getattr(module_of(test_id), class_name)(method)]).run(result)
    return result


def passed(result) -> bool:
    return not (result.errors or result.failures or result.unexpectedSuccesses)


def silence_stdout() -> None:
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)


def fan_out(tasks, parallel: int, alarm_s: int, report_every: int = 100) -> None:
    """tasks: [(output path, work(output path))]. Each runs alone in a fresh fork, ``parallel``
    at a time; a task whose output already exists is skipped (runs can be resumed)."""
    queue = list(tasks)
    running: dict[int, str] = {}
    began, finished = time.time(), 0
    while queue or running:
        while queue and len(running) < parallel:
            output, work = queue.pop(0)
            if os.path.exists(output):
                continue
            sys.stdout.flush()
            sys.stderr.flush()
            pid = os.fork()
            if pid == 0:
                try:
                    signal.alarm(alarm_s)
                    silence_stdout()
                    work(output)
                except BaseException as problem:  # noqa
                    os.write(2, f"{output}: {problem!r}\n".encode())
                os._exit(0)
            running[pid] = output
        if not running:
            break
        pid, _ = os.wait()
        running.pop(pid, None)
        finished += 1
        if report_every and finished % report_every == 0:
            print(f"{finished} done in {time.time() - began:.0f}s", file=sys.stderr, flush=True)
    print(f"all done in {time.time() - began:.0f}s", file=sys.stderr, flush=True)


def heaviest_first(tests: list[dict]) -> list[dict]:
    return sorted(tests, key=lambda t: -t["seconds"])


def write_atomically(path: str, write) -> None:
    partial = path + ".partial"
    with open(partial, "wb") as handle:
        write(handle)
    os.rename(partial, path)


def by_file(pairs) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for path, line in pairs:
        grouped.setdefault(path, []).append(line)
    return {path: sorted(lines) for path, lines in grouped.items()}


@contextmanager
def recording_lines(prefix: str = PREFIX, tool: int = 3, name: str = "forge-lines", skip_imports: bool = False):
    """Collect (path relative to the pinned root, line) for every line run under ``prefix``.
    Each line is reported once (the callback disables itself per location). With
    ``skip_imports``, lines run by an import are left live and not recorded."""
    monitoring = sys.monitoring
    seen: set[tuple[str, int]] = set()
    root_length = len(PREFIX)

    def on_line(code, line):
        file_name = code.co_filename
        if file_name.startswith(prefix):
            if skip_imports and _inside_import():
                return None
            seen.add((file_name[root_length:], line))
        return monitoring.DISABLE

    monitoring.use_tool_id(tool, name)
    monitoring.register_callback(tool, monitoring.events.LINE, on_line)
    monitoring.set_events(tool, monitoring.events.LINE)
    try:
        yield seen
    finally:
        monitoring.set_events(tool, 0)
        monitoring.register_callback(tool, monitoring.events.LINE, None)
        monitoring.free_tool_id(tool)


def _inside_import() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_filename.startswith("<frozen importlib"):
            return True
        frame = frame.f_back
    return False


class ClassCreations:
    """Wraps ``__init__`` of every sqlglot class so creations are noted while ``active``."""

    def __init__(self):
        self.active: set | None = None

    def install(self) -> None:
        import functools
        import inspect

        seen = set()
        for name, module in list(sys.modules.items()):
            if not name.startswith("sqlglot"):
                continue
            for value in list(vars(module).values()):
                if not (inspect.isclass(value) and value.__module__.startswith("sqlglot")) or value in seen:
                    continue
                seen.add(value)
                init = value.__dict__.get("__init__")
                if init is None or not inspect.isfunction(init):
                    continue
                try:
                    value.__init__ = self._noting(init, functools)
                except Exception:  # noqa
                    pass

    def _noting(self, init, functools):
        @functools.wraps(init)
        def __init__(instance, *args, **kwargs):
            if self.active is not None:
                self.active.add(type(instance))
            return init(instance, *args, **kwargs)

        return __init__

    @staticmethod
    def names(classes) -> list[str]:
        """'module:qualname' of the classes and their sqlglot bases."""
        return sorted({f"{k.__module__}:{k.__qualname__}" for c in classes for k in c.__mro__
                       if k.__module__.startswith("sqlglot")})
