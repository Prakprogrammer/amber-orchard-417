"""Measure ripple against ground-truth labels (``forge.labels``).

    python -m forge.evaluate selection <data dir> <labels.json>
        recall of the candidates alone (no runs): misses, and what listing them all would cost
    python -m forge.evaluate output <data dir> <labels.json> <ripple output.jsonl>
        the final answer: commits with a missed failing test, and the listed seconds
"""

from __future__ import annotations

import json
import os
import sys

from ripple import bits
from ripple.costs import listing_seconds


def _load(data_dir: str, labels_path: str):
    with open(os.path.join(data_dir, "tests.json")) as handle:
        tests = json.load(handle)["tests"]
    with open(labels_path) as handle:
        labels = json.load(handle)["commits"]
    return tests, [round(t["seconds"] * 1000) for t in tests], labels


def selection(data_dir: str, labels_path: str) -> None:
    from ripple.diffs import read_diff
    from ripple.pipeline import Pipeline
    from ripple.settings import Settings

    os.environ["RIPPLE_DATA"] = data_dir
    pipeline = Pipeline(Settings.from_environment())
    pipeline.load()
    tests, durations, labels = _load(data_dir, labels_path)
    misses, total = 0, 0.0
    for commit in sorted(labels):
        with open(os.path.join(data_dir, "commits", commit + ".patch"), encoding="utf-8", newline="") as handle:
            impact = pipeline.analyzer.analyze(read_diff(handle.read()))
        chosen = set(range(len(tests))) if impact.everything else set(bits.members(impact.candidates))
        missed = set(labels[commit]["failing"]) - chosen
        misses += bool(missed)
        total += listing_seconds(chosen, durations)
        print(f"{commit} failing={len(labels[commit]['failing']):3d} candidates={len(chosen):4d} "
              + (f"MISSED {[tests[n]['id'].split('::')[-1] for n in missed]}" if missed else ""))
    print(f"commits with a miss: {misses}; listing every candidate: {total:.1f}s")


def output(data_dir: str, labels_path: str, answer_path: str) -> None:
    tests, durations, labels = _load(data_dir, labels_path)
    answers = {}
    with open(answer_path) as handle:
        for row in handle:
            if row.strip():
                record = json.loads(row)
                answers[record["commit"]] = {n for worker in record["workers"] for n in worker}
    misses, total = 0, 0.0
    for commit in sorted(labels):
        listed = answers.get(commit, set())
        missed = set(labels[commit]["failing"]) - listed
        misses += bool(missed)
        total += listing_seconds(listed, durations)
        if missed:
            print(f"{commit}: MISSED {[tests[n]['id'] for n in sorted(missed)][:8]}")
    print(f"commits with a miss: {misses} of {len(labels)}; listed {total:.1f}s")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    selection(*args) if command == "selection" else output(*args)
