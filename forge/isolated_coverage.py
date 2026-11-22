"""Per-test coverage with every test alone in a fresh fork of a fully imported parent, so lazily
built caches (generator dispatch tables, per-dialect state...) are rebuilt by each test that
needs them and show up in its coverage. Same record format as ``pytest_coverage``, plus the
directories each test listed.

    python -m forge.isolated_coverage /data/tests.json <out dir> [parallel]
"""

from __future__ import annotations

import json
import os
import sys

from forge import common


class _FileWatch:
    def __init__(self):
        self.opened: set | None = None
        self.listed: set | None = None

    def __call__(self, event, args):
        if self.opened is None:
            return
        path = args[0] if args else None
        if isinstance(path, bytes):
            path = path.decode(errors="replace")
        if not isinstance(path, str):
            return
        if event == "open":
            resolved = path if path.startswith(common.PREFIX) else os.path.realpath(path)
            if resolved.startswith(common.PREFIX):
                if resolved.endswith((".py", ".pyc")) and common._inside_import():
                    return
                if not resolved.endswith(".pyc"):
                    self.opened.add(resolved[len(common.PREFIX):])
        elif event in ("os.listdir", "os.scandir"):
            resolved = os.path.realpath(path)
            if resolved.startswith(common.PREFIX):
                self.listed.add(resolved[len(common.PREFIX):])


def main(tests_json: str, out_dir: str, parallel: int = 4) -> None:
    common.enter_pinned_tree()
    os.makedirs(out_dir, exist_ok=True)
    tests = common.load_tests(tests_json)
    common.calm_the_process()
    common.import_everything(tests)
    creations = common.ClassCreations()
    creations.install()
    watch = _FileWatch()
    sys.addaudithook(watch)

    def record(test):
        def work(output):
            creations.active, watch.opened, watch.listed = set(), set(), set()
            with common.recording_lines(prefix=common.PREFIX) as lines:
                result = common.run_alone(test["id"])
            row = {"id": test["id"], "lines": common.by_file(lines), "opens": sorted(watch.opened),
                   "dirs": sorted(watch.listed), "classes": creations.names(creations.active), "ok": common.passed(result)}
            with open(output, "w") as handle:
                handle.write(json.dumps(row) + "\n")
        return work

    tasks = [(os.path.join(out_dir, f"iso-{t['number']}.jsonl"), record(t)) for t in common.heaviest_first(tests)]
    common.fan_out(tasks, parallel, alarm_s=900)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
