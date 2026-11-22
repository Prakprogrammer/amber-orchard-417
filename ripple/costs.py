"""What listing and running tests cost, and how listed tests are split over workers.

Listing a test costs its recorded seconds inside one of up to eight parallel workers (plus a
fixed 2 s per worker). Running it here first costs our own CPU; afterwards only the tests seen
failing are listed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .atlas import test_class, test_module

WORKER_OVERHEAD_MS = 2000
MAX_WORKERS = 8
# running any test of these makes the parent import numpy / pandas / duckdb first
_PANDAS_MODULES = frozenset({"tests.test_executor", "tests.test_optimizer"})
_PANDAS_IMPORT_S = 2.0
_START_FRESH_S = 0.35  # fork + a fresh import of the patched tree
_START_DATA_S = 0.12  # fork, no python change


@dataclass
class Packing:
    cost_ms: int
    workers: list[list[int]]
    loads: list[int]


def _longest_first(tests: list[int], durations: list[int], worker_count: int):
    loads = [0] * worker_count
    workers: list[list[int]] = [[] for _ in range(worker_count)]
    for test in sorted(tests, key=lambda t: -durations[t]):
        lightest = loads.index(min(loads))
        loads[lightest] += durations[test]
        workers[lightest].append(test)
    return loads, workers


def _rebalance(workers: list[list[int]], loads: list[int], durations: list[int]) -> None:
    """Local search after LPT: move or swap tests off the busiest worker while that lowers it."""
    for _ in range(200):
        top = loads.index(max(loads))
        best = None  # (new peak of the pair, kind, ...)
        for other in range(len(workers)):
            if other == top:
                continue
            for i, test in enumerate(workers[top]):
                d = durations[test]
                peak = max(loads[top] - d, loads[other] + d)
                if peak < loads[top] and (best is None or peak < best[0]):
                    best = (peak, "move", other, i, None)
                for j, swap in enumerate(workers[other]):
                    e = durations[swap]
                    if e >= d:
                        continue
                    peak = max(loads[top] - d + e, loads[other] + d - e)
                    if peak < loads[top] and (best is None or peak < best[0]):
                        best = (peak, "swap", other, i, j)
        if best is None:
            return
        _, kind, other, i, j = best
        test = workers[top].pop(i)
        loads[top] -= durations[test]
        if kind == "swap":
            swap = workers[other].pop(j)
            loads[other] -= durations[swap]
            workers[top].append(swap)
            loads[top] += durations[swap]
        workers[other].append(test)
        loads[other] += durations[test]


def pack(tests, durations: list[int]) -> Packing:
    """The cheapest longest-processing-time split over 1..8 workers. Equal durations keep the
    order the tests come in (a set's own order), which decides who shares a worker."""
    ordered = list(tests)
    if not ordered:
        return Packing(0, [], [])
    best = None
    for worker_count in range(1, min(MAX_WORKERS, len(ordered)) + 1):
        loads, workers = _longest_first(ordered, durations, worker_count)
        if worker_count > 1 and len(ordered) <= 400:
            _rebalance(workers, loads, durations)
        cost = max(loads) + WORKER_OVERHEAD_MS * worker_count
        if best is None or cost < best.cost_ms:
            best = Packing(cost, workers, loads)
    return best


def listing_seconds(tests, durations: list[int]) -> float:
    return pack(tests, durations).cost_ms / 1000


@dataclass
class RunPricing:
    """Per-commit inputs of a run's price."""

    fresh_import: bool = True  # False: data-only change, nothing to re-import
    hot_swap: bool = False  # function bodies swapped into the preloaded modules
    after_run_s: float = 1.0  # expected cost of listing what the run finds failing
    reduced: dict[int, float] = field(default_factory=dict)  # test -> CPU of its reduced run


class CostModel:
    def __init__(self, tests: list[dict], costs: dict):
        self.ids = [t["id"] for t in tests]
        self.seconds = [t["seconds"] for t in tests]
        self.durations_ms = [round(t["seconds"] * 1000) for t in tests]
        measured = costs.get("test_cpu", {})
        self.run_cpu = [measured.get(i, 2.0 * s + 0.02) for i, s in zip(self.ids, self.seconds)]
        self.class_of = [test_class(i) for i in self.ids]
        self.module_of = [test_module(i) for i in self.ids]
        self.class_setup_cpu: dict[str, float] = costs.get("class_setup_cpu", {})
        self.module_import_cpu: dict[str, float] = costs.get("module_import_cpu", {})
        # the scoring machine ran ~2-5x faster than the one the costs were measured on
        self.cpu_scale = float(os.environ.get("RIPPLE_CPU_SCALE", "0.5"))
        # tests whose run alone would take more than this are never run (pool-heavy tests)
        self.max_run_s = float(os.environ.get("RIPPLE_MAX_RUN_S", "30.0"))
        self.swap_start_s = float(os.environ.get("RIPPLE_SWAP_START_S", "0.1"))
        self.too_costly = {i for i, cpu in enumerate(self.run_cpu) if cpu * self.cpu_scale > self.max_run_s}

    def test_cpu(self, test: int, pricing: RunPricing) -> float:
        """Our CPU seconds to run the test (reduced, when the commit reduces it)."""
        return pricing.reduced.get(test, self.run_cpu[test]) * self.cpu_scale

    def run_cost(self, tests, pricing: RunPricing) -> float:
        if not tests:
            return 0.0
        if pricing.hot_swap:
            cost = self.swap_start_s
        else:
            cost = _START_FRESH_S if pricing.fresh_import else _START_DATA_S
        cost += pricing.after_run_s
        classes, modules = set(), set()
        for test in tests:
            cost += self.test_cpu(test, pricing)
            classes.add(self.class_of[test])
            modules.add(self.module_of[test])
        for klass in classes:
            cost += self.class_setup_cpu.get(klass, 0.0) * self.cpu_scale
        if modules & _PANDAS_MODULES:
            cost += _PANDAS_IMPORT_S * self.cpu_scale
        if not pricing.hot_swap:
            for module in modules:
                cost += self.module_import_cpu.get(module, 0.0) * self.cpu_scale
        return cost
