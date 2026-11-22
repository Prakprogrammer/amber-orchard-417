"""Ground truth: which tests a patch really breaks, by running pytest on the patched tree.

    python -m forge.labels <data dir> <tests.json> <out.json> [commit ...]

Runs the whole suite per commit, or, with $FORGE_ONLY_CANDIDATES=<cands.json> (as written by
``forge.mutants comparisons``), only each commit's listed candidates. Used for evaluation only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from forge import common
from ripple.diffs import patch_text, read_diff


def _patched_copy(patch: str) -> str:
    work = tempfile.mkdtemp(prefix="forge-label-")
    tree = os.path.join(work, "tree")
    shutil.copytree(common.PINNED_ROOT, tree, symlinks=True)
    for diff in read_diff(patch):
        old = None
        if diff.source:
            with open(os.path.join(tree, diff.source), encoding="utf-8", newline="") as handle:
                old = handle.read()
            os.remove(os.path.join(tree, diff.source))
        new = patch_text(old, diff)
        if new is not None:
            target = os.path.join(tree, diff.target)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="") as handle:
                handle.write(new)
    return work


def main(data_dir: str, tests_json: str, output: str, *only: str) -> None:
    tests = common.load_tests(tests_json)
    number_of = {t["id"]: t["number"] for t in tests}
    numbers_in_file: dict[str, list[int]] = {}
    for t in tests:
        numbers_in_file.setdefault(t["file"], []).append(t["number"])
    candidates = None
    if os.environ.get("FORGE_ONLY_CANDIDATES"):
        with open(os.environ["FORGE_ONLY_CANDIDATES"]) as handle:
            candidates = json.load(handle)
    commits = sorted(n[:-6] for n in os.listdir(os.path.join(data_dir, "commits")) if n.endswith(".patch"))
    labels = {}
    for commit in commits:
        if (only and commit not in only) or (candidates is not None and commit not in candidates):
            continue
        with open(os.path.join(data_dir, "commits", commit + ".patch"), encoding="utf-8", newline="") as handle:
            work = _patched_copy(handle.read())
        failures = os.path.join(work, "failures.txt")
        env = dict(os.environ, PYTHONPATH=f"{work}/tree:{common.REPO}", FORGE_FAILURES=failures, SKIP_INTEGRATION="0")
        targets = candidates[commit]["candidates"] if candidates is not None else ["tests"]
        began = time.time()
        finished = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "forge.pytest_failures", "-p", "no:cacheprovider",
                                   "-n", os.environ.get("FORGE_WORKERS", "4"), "--timeout", os.environ.get("FORGE_TIMEOUT", "300")]
                                  + targets, cwd=f"{work}/tree", env=env, capture_output=True, text=True)
        failing, unknown = set(), []
        if os.path.exists(failures):
            with open(failures) as handle:
                for row in handle:
                    kind, node = row.rstrip("\n").split("\t")
                    if kind == "test" and node in number_of:
                        failing.add(number_of[node])
                    elif kind == "collect" and node.split("::")[0] in numbers_in_file:
                        failing.update(numbers_in_file[node.split("::")[0]])
                    else:
                        unknown.append(f"{kind}:{node}")
        labels[commit] = {"failing": sorted(failing), "unknown": unknown, "seconds": round(time.time() - began, 1),
                          "summary": finished.stdout.strip().splitlines()[-1:]}
        print(commit, labels[commit], file=sys.stderr, flush=True)
        shutil.rmtree(work, ignore_errors=True)
        with open(output, "w") as handle:
            json.dump({"commits": labels}, handle, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
