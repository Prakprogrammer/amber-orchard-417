"""pytest plugin: append "<kind>\\t<node id>" for each failed test (subtests and setup errors
included) or failed collection to $FORGE_FAILURES."""

import os

_OUTPUT = os.environ.get("FORGE_FAILURES", "/tmp/forge-failures.txt")
_written = set()


def _note(kind, node_id):
    if (kind, node_id) not in _written:
        _written.add((kind, node_id))
        with open(_OUTPUT, "a") as handle:
            handle.write(f"{kind}\t{node_id}\n")


def pytest_runtest_logreport(report):
    if report.failed:
        _note("test", report.nodeid)


def pytest_collectreport(report):
    if report.failed:
        _note("collect", report.nodeid)
