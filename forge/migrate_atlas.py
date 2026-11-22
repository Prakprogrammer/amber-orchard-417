"""Convert map files of the earlier layout (map/*.pkl.gz, map/runcost.json) to the atlas.

    python -m forge.migrate_atlas <old map dir> <atlas dir>

Only the layout changes (names of fields, typed pass labels, one untrusted mask); every
recorded fact is carried over as it is, so no test run is needed.
"""

from __future__ import annotations

import json
import os
import sys

from ripple.atlas import read_artifact, write_artifact

GENERATOR_MEMORY = "tests/test_generator.py::TestGenerator::test_generator_memory"
_LOADER_KINDS = {"load_sql_fixture_pairs": "pairs", "load_sql_fixtures": "lines"}


def coverage(old: dict) -> dict:
    return {
        "test_ids": old["ids"],
        "line_hits": old["lines"],
        "file_reads": old["opens"],
        "dir_scans": old.get("dirs", {}),
        "instantiated": old["classes"],
        "module_loads": old.get("modsets", {}),
        "import_time_lines": {path: set(lines) for path, lines in old["import_lines"].items()},
        "import_time_calls": {
            path: [(first, last, tuple(tuple(body) for body in bodies)) for first, last, bodies in spans]
            for path, spans in old.get("import_consumers", {}).items()},
        "imported_by": old.get("importers", {}),
        "lazy_imports": {path: [tuple(site) for site in sites] for path, sites in old.get("func_imports", {}).items()},
        "name_reads": {name: [tuple(site) for site in sites] for name, sites in old.get("attr_refs", {}).items()},
        "dynamic_imports": {prefix: [tuple(site) for site in sites] for prefix, sites in old.get("dyn_imports", {}).items()},
    }


def comparisons(old: dict) -> dict:
    return {
        "test_ids": old["ids"],
        "sites": old["sites"],
        "import_time": set(old["parent"]),
        "results": old["sig"],
        "operands": old["vals"],
        "operands_lost": old["vals_unk"],
        "untrusted": old["bad"] | old["missing"],
        "files": sorted(old["files"]),
    }


def pass_label(test_id: str, key: str) -> tuple:
    if "|" in key:
        loader, fixture, index = key.split("|")
        return (_LOADER_KINDS[loader], fixture, int(index))
    if test_id == GENERATOR_MEMORY:
        return ("dialect", key)
    return ("query", int(key))


def passes(old: dict) -> dict:
    """Bit i of every reach mask still means labels[i]: the label order is the old key order."""
    return {test_id: None if entry is None else {
        "labels": [pass_label(test_id, key) for key in entry["passes"]],
        "reach": entry["lines"],
    } for test_id, entry in old.items()}


def tpcds(old: dict) -> list:
    renamed = {"sql": "text", "duck": "duckdb_text", "opt": "optimized", "rows": "rows", "late": "executor_lines",
               "execute": "execute", "stable": "stable"}
    return [None if query is None else {renamed[k]: v for k, v in query.items()} for query in old["queries"]]


def costs(old: dict) -> dict:
    return {"test_cpu": old.get("test", {}), "class_setup_cpu": old.get("setup", {}),
            "module_import_cpu": old.get("modtime", {})}


def main(old_dir: str, atlas_dir: str) -> None:
    os.makedirs(atlas_dir, exist_ok=True)
    conversions = [("covmap.pkl.gz", "coverage.pkl.gz", coverage, 9), ("outcomes.pkl.gz", "comparisons.pkl.gz", comparisons, 6),
                   ("passmap.pkl.gz", "passes.pkl.gz", passes, 6), ("tpcds_stage.pkl.gz", "tpcds.pkl.gz", tpcds, 6)]
    for old_name, new_name, convert, level in conversions:
        source = os.path.join(old_dir, old_name)
        if os.path.exists(source):
            write_artifact(os.path.join(atlas_dir, new_name), convert(read_artifact(source)), level)
            print(f"{old_name} -> {new_name}", file=sys.stderr)
    cost_file = os.path.join(old_dir, "runcost.json")
    if os.path.exists(cost_file):
        with open(cost_file) as handle:
            table = costs(json.load(handle))
        with open(os.path.join(atlas_dir, "costs.json"), "w") as handle:
            json.dump(table, handle, indent=0, sort_keys=True)
        print("runcost.json -> costs.json", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:3])
