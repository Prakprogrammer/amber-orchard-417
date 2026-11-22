"""Which sqlglot and test modules each test loads, run alone in a fresh fork of a parent that
has imported no sqlglot module yet (so lazy loaders, dynamic imports and import chains all show
up). A module-level edit of module M, or an import failure in it, can only reach tests that
loaded M.

    python -m forge.module_loads /data/tests.json <out.json> [parallel]
"""

from __future__ import annotations

import json
import os
import sys
import time

from forge import common


def main(tests_json: str, output: str, parallel: int = 4) -> None:
    common.enter_pinned_tree()
    tests = common.load_tests(tests_json)
    common.calm_the_process()
    for name in ("duckdb", "pandas", "pandas.testing"):  # third-party only: no sqlglot yet
        __import__(name)
    assert not any(m == "sqlglot" or m.startswith("sqlglot.") for m in sys.modules)
    scratch = "/tmp/forge-module-loads"
    os.makedirs(scratch, exist_ok=True)

    def record(test):
        def work(path):
            import importlib

            importlib.import_module(test["file"][:-3].replace("/", "."))
            common.run_alone(test["id"])
            loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("sqlglot", "tests"))
            with open(path, "w") as handle:
                json.dump({"id": test["id"], "modules": loaded}, handle)
        return work

    began = time.time()
    common.fan_out([(os.path.join(scratch, f"{t['number']}.json"), record(t)) for t in common.heaviest_first(tests)],
                   parallel, alarm_s=900, report_every=0)
    loads = {}
    for test in tests:
        path = os.path.join(scratch, f"{test['number']}.json")
        if os.path.exists(path):
            with open(path) as handle:
                row = json.load(handle)
            loads[row["id"]] = row["modules"]
    with open(output, "w") as handle:
        json.dump(loads, handle)
    print(f"{len(loads)}/{len(tests)} tests in {time.time() - began:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
