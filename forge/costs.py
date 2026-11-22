"""Measure what running each test costs in ripple's own runner on the pinned tree (CPU seconds):
per test, per class setup, per test-module import. Every class runs in its own child that
re-imports sqlglot, as for a commit editing python code.

    python -m forge.costs /data/tests.json atlas/costs.json        (inside the image)
"""

from __future__ import annotations

import json
import os
import sys

from forge import common


def main(tests_json: str, output: str) -> None:
    from ripple.harness import ChildOrders
    from ripple.pipeline import _preload_libraries, log
    from ripple.scheduler import Job, run_all

    tests = common.load_tests(tests_json)
    _preload_libraries(True, common.PINNED_ROOT)
    by_class: dict[str, list[str]] = {}
    for test in tests:
        by_class.setdefault(test["id"].rsplit("::", 1)[0], []).append(test["id"])
    jobs = [Job(f"class{i}", ChildOrders(tree=f"/tmp/forge-costs/class{i}", pinned_root=common.PINNED_ROOT, test_ids=ids,
                                         cpu_limits={t: 600.0 for t in ids}, reimport_sqlglot=True), {})
            for i, ids in enumerate(by_class.values())]
    run_all(jobs, int(os.environ.get("RIPPLE_PARALLEL", "2")), None, log)
    table = {"test_cpu": {}, "class_setup_cpu": {}, "module_import_cpu": {}}
    not_passing = {}
    for job in jobs:
        for test_id, report in job.verdicts.items():
            table["test_cpu"][test_id] = report["cpu"]
            if report["verdict"] != "pass":
                not_passing[test_id] = report["verdict"]
        for key, cpu in job.timings.items():
            kind, name = key.split(":", 1)
            table[f"{kind}_cpu"][name] = cpu
    print(f"measured {len(table['test_cpu'])} tests; not passing: {not_passing}", file=sys.stderr)
    with open(output, "w") as handle:
        json.dump(table, handle, indent=0, sort_keys=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
