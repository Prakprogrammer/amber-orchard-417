"""For every function that runs while sqlglot (and the test modules) are imported: which class or
module body line it ran for (the innermost body executing when it was called, at that body's
current line). An edit inside such a function only reaches what those body lines built.

    python -m forge.import_trace <out.json> [/data/tests.json]

Output: {path: [[first line, last line, [[body path, body qualname, body line], ...]], ...]}
"""

from __future__ import annotations

import json
import sys

from forge import common

_CO_OPTIMIZED = 0x0001  # functions; module and class bodies lack it
TOOL = 3


def main(output: str, tests_json: str | None = None) -> None:
    common.enter_pinned_tree()
    monitoring = sys.monitoring
    events = monitoring.events
    root_length = len(common.PREFIX)
    open_bodies: list[tuple[str, str, object]] = []  # (path, qualname, frame) of running bodies
    calls: dict = {}  # function code -> {(body path, body qualname, body line)}

    def on_start(code, offset):
        if not code.co_filename.startswith(common.PREFIX):
            return None
        if not code.co_flags & _CO_OPTIMIZED:
            name = "<module>" if code.co_name == "<module>" else code.co_qualname
            open_bodies.append((code.co_filename[root_length:], name, sys._getframe(1)))
        elif open_bodies:
            path, name, frame = open_bodies[-1]
            calls.setdefault(code, set()).add((path, name, frame.f_lineno))
        return None

    def on_end(code, offset, *rest):
        if code.co_filename.startswith(common.PREFIX) and not code.co_flags & _CO_OPTIMIZED and open_bodies:
            open_bodies.pop()
        return None

    monitoring.use_tool_id(TOOL, "forge-import-trace")
    monitoring.register_callback(TOOL, events.PY_START, on_start)
    monitoring.register_callback(TOOL, events.PY_RETURN, on_end)
    monitoring.register_callback(TOOL, events.PY_UNWIND, on_end)
    monitoring.set_events(TOOL, events.PY_START | events.PY_RETURN | events.PY_UNWIND)
    common.import_everything(common.load_tests(tests_json) if tests_json else [], strict_tests=False)
    monitoring.set_events(TOOL, 0)

    spans: dict[str, list] = {}
    for code, bodies in calls.items():
        lines = [code.co_firstlineno] + [line for _, _, line in code.co_lines() if line is not None]
        spans.setdefault(code.co_filename[root_length:], []).append([min(lines), max(lines), sorted(list(b) for b in bodies)])
    with open(output, "w") as handle:
        json.dump(spans, handle)
    print(f"{len(spans)} files, {sum(len(v) for v in spans.values())} functions", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:3])
