"""pytest plugin: per test, the sqlglot/test lines it ran, the files it opened and the sqlglot
classes it created. One JSON record per test in $FORGE_COVERAGE_DIR/tests-<pid>.jsonl, plus a
record "<import>" with the lines run before any test (imports, collection).

Child processes (spawned pools, subprocesses) are covered by ``sitecustomize.py``: put the
forge directory on PYTHONPATH. Pool workers outlive a test, so every task submitted to a pool
is wrapped in a ``TaggedTask`` naming the test that sent it.

    python -m pytest -p forge.pytest_coverage -p no:cacheprovider -n 8 tests
"""

import json
import os
import sys

import pytest

ROOT = os.path.realpath(os.environ.get("FORGE_COVERAGE_ROOT", os.getcwd())) + "/"
OUTPUT_DIR = os.environ.get("FORGE_COVERAGE_DIR", "")
TOOL = 3
monitoring = sys.monitoring


class _Recording:
    def __init__(self):
        self.sink = None  # where lines go right now (None: nowhere)
        self.import_lines: set = set()
        self.phases: dict | None = None
        self.opened: set | None = None
        self.output = None


STATE = _Recording()
_CLASSES = None


def _on_line(code, line):
    file_name = code.co_filename
    if file_name.startswith(ROOT) and STATE.sink is not None:
        STATE.sink.add((file_name[len(ROOT):], line))
    return monitoring.DISABLE


def _on_audit(event, args):
    if event != "open" or STATE.opened is None:
        return
    path = args[0]
    if isinstance(path, bytes):
        path = path.decode(errors="replace")
    if not isinstance(path, str):
        return
    resolved = path if path.startswith(ROOT) else os.path.realpath(path)
    if resolved.startswith(ROOT) and not resolved.endswith((".py", ".pyc")):
        STATE.opened.add(resolved[len(ROOT):])


def _grouped(pairs) -> dict:
    grouped: dict = {}
    for path, line in pairs:
        grouped.setdefault(path, []).append(line)
    return {path: sorted(lines) for path, lines in grouped.items()}


def pytest_configure(config):
    controller = (getattr(config, "workerinput", None) is None and os.environ.get("PYTEST_XDIST_WORKER") is None
                  and config.option.numprocesses not in (None, 0))
    if controller:
        return  # with xdist, the workers record
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    STATE.output = open(os.path.join(OUTPUT_DIR, f"tests-{os.getpid()}.jsonl"), "w")
    monitoring.use_tool_id(TOOL, "forge-coverage")
    monitoring.register_callback(TOOL, monitoring.events.LINE, _on_line)
    STATE.sink = STATE.import_lines
    monitoring.set_events(TOOL, monitoring.events.LINE)
    sys.addaudithook(_on_audit)


def pytest_collection_finish(session):
    global _CLASSES
    if STATE.output is None:
        return
    from forge.common import ClassCreations, import_everything

    import_everything()
    _CLASSES = ClassCreations()
    _CLASSES.install()
    _tag_pool_tasks()


def _tag_pool_tasks():
    import concurrent.futures.process as process_futures

    import sitecustomize

    submit = process_futures.ProcessPoolExecutor.submit

    def tagged_submit(self, fn, /, *args, **kwargs):
        test_id = os.environ.get("FORGE_COVERAGE_TEST")
        if test_id:
            fn = sitecustomize.TaggedTask(fn, test_id)
        return submit(self, fn, *args, **kwargs)

    process_futures.ProcessPoolExecutor.submit = tagged_submit


def pytest_sessionfinish(session):
    if STATE.output is not None:
        STATE.output.write(json.dumps({"id": "<import>", "lines": _grouped(STATE.import_lines)}) + "\n")
        STATE.output.flush()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if STATE.output is None:
        yield
        return
    os.environ["FORGE_COVERAGE_TEST"] = item.nodeid
    STATE.phases = {"setup": set(), "call": set(), "teardown": set()}
    STATE.opened = set()
    if _CLASSES is not None:
        _CLASSES.active = set()
    monitoring.restart_events()
    yield
    STATE.sink = None
    created = _CLASSES.active if _CLASSES is not None else set()
    STATE.output.write(json.dumps({
        "id": item.nodeid,
        "setup": _grouped(STATE.phases["setup"]),
        "lines": _grouped(STATE.phases["call"] | STATE.phases["teardown"]),
        "opens": sorted(STATE.opened),
        "classes": _CLASSES.names(created) if _CLASSES is not None else [],
    }) + "\n")
    STATE.output.flush()
    STATE.opened = None
    if _CLASSES is not None:
        _CLASSES.active = None


def _phase(name):
    @pytest.hookimpl(hookwrapper=True)
    def hook(item):
        if STATE.output is not None and STATE.phases is not None:
            STATE.sink = STATE.phases[name]
        yield

    return hook


pytest_runtest_setup = _phase("setup")
pytest_runtest_call = _phase("call")
pytest_runtest_teardown = _phase("teardown")
