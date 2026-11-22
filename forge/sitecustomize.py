"""Loaded by every Python process started while the coverage is recorded (the forge directory
is on PYTHONPATH). In child processes (spawned pools, subprocesses) it records the lines under
$FORGE_COVERAGE_ROOT for the test running, and writes them to sub-<pid>.jsonl on exit."""

import os
import sys

_RECORDING = bool(os.environ.get("FORGE_COVERAGE_DIR") and os.environ.get("FORGE_COVERAGE_TEST"))
_lines_by_test: dict = {}
_current = None


class TaggedTask:
    """A pool task that names the test it belongs to (workers are reused across tests)."""

    def __init__(self, fn, test_id):
        self.fn = fn
        self.test_id = test_id

    def __call__(self, *args, **kwargs):
        global _current
        if _RECORDING:
            _current = _lines_by_test.setdefault(self.test_id, set())
            sys.monitoring.restart_events()
        return self.fn(*args, **kwargs)


if _RECORDING:
    import atexit
    import json

    _ROOT = os.path.realpath(os.environ.get("FORGE_COVERAGE_ROOT", os.getcwd())) + "/"
    _current = _lines_by_test.setdefault(os.environ["FORGE_COVERAGE_TEST"], set())
    _monitoring = sys.monitoring

    def _on_line(code, line):
        if code.co_filename.startswith(_ROOT) and _current is not None:
            # a worker imports sqlglot afresh: lines run by an import are not the test's doing
            frame = sys._getframe(1)
            while frame is not None:
                if frame.f_code.co_filename.startswith("<frozen importlib"):
                    return None  # stay live: the line may run again outside the import
                frame = frame.f_back
            _current.add((code.co_filename[len(_ROOT):], line))
        return _monitoring.DISABLE

    _monitoring.use_tool_id(4, "forge-coverage-child")
    _monitoring.register_callback(4, _monitoring.events.LINE, _on_line)
    _monitoring.set_events(4, _monitoring.events.LINE)

    def _write_out():
        with open(os.path.join(os.environ["FORGE_COVERAGE_DIR"], f"sub-{os.getpid()}.jsonl"), "w") as handle:
            for test_id, lines in _lines_by_test.items():
                grouped: dict = {}
                for path, line in lines:
                    grouped.setdefault(path, []).append(line)
                handle.write(json.dumps({"id": test_id, "lines": grouped}) + "\n")

    atexit.register(_write_out)
