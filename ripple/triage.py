"""Per commit: which candidates to run here first, which to list outright, which to hold.

- cheap to run (relative to listing): run, and list only if it fails;
- costly to run: list;
- long tests (above ``held_above_s``) whose run would cost too much are *held*: they are listed
  only on evidence, i.e. when a run test of their class (or one reading their fixture files)
  fails. The candidates that can give that evidence are *shielded*: they run even when the
  commit's CPU budget would leave them out.
"""

from __future__ import annotations

from dataclasses import dataclass

from .atlas import CoverageAtlas, test_class
from .costs import CostModel, RunPricing
from .settings import Settings


@dataclass
class Triage:
    listed: set[int]
    run: set[int]
    held: set[int]
    shielded: set[int]
    budget_cut: bool  # some runnable tests were listed to stay inside the commit's budget


class FixtureIndex:
    """The data files (not .py) each test opened."""

    def __init__(self, atlas: CoverageAtlas):
        self.atlas = atlas
        self._by_test: dict[int, set[str]] | None = None

    def of(self, test: int) -> set[str]:
        if self._by_test is None:
            from . import bits

            self._by_test = {}
            for path, readers in self.atlas.file_reads.items():
                if not path.endswith(".py"):
                    for reader in bits.members(readers):
                        self._by_test.setdefault(reader, set()).add(path)
        return self._by_test.get(test, set())


def triage(candidates: set[int], data_readers: set[int], hot_swap: bool, pricing: RunPricing, dialect_reduced: set[int],
           model: CostModel, tests: list[dict], fixtures: FixtureIndex, settings: Settings) -> Triage:
    def cpu(test: int) -> float:
        return model.test_cpu(test, pricing)

    seconds = [t["seconds"] for t in tests]
    listed = set(data_readers & candidates)
    held, runnable = set(), set()
    for test in candidates - data_readers:
        cost = cpu(test)
        if seconds[test] > settings.held_above_s and cost > settings.held_run_share * seconds[test]:
            held.add(test)
        elif test in model.too_costly and test not in pricing.reduced:
            listed.add(test)
        elif cost <= settings.always_run_below_s or cost <= settings.run_ratio * (seconds[test] + 0.3):
            runnable.add(test)
        elif test in dialect_reduced and cost <= settings.dialect_run_ratio * (seconds[test] + 0.3):
            runnable.add(test)  # a few fixture entries: running checks what listing would
        else:
            listed.add(test)

    shielded = set()
    if held:
        held_classes = {test_class(tests[t]["id"]) for t in held}
        held_fixtures = set().union(*(fixtures.of(t) for t in held))
        shielded = {t for t in runnable if test_class(tests[t]["id"]) in held_classes
                    or (held_fixtures and held_fixtures & fixtures.of(t) and cpu(t) <= 2.0)}
    shielded |= runnable & dialect_reduced  # these run outside the commit's budget

    budget = settings.swap_commit_budget_s if hot_swap else settings.fresh_commit_budget_s
    if sum(cpu(t) for t in runnable - dialect_reduced) <= budget:
        return Triage(listed, runnable, held, shielded, False)
    run, spent = set(), 0.0
    # shielded first, then the most listing time saved per CPU second
    for test in sorted(runnable, key=lambda t: (t not in shielded, -(seconds[t] + 0.002) / (cpu(t) + 0.002))):
        if spent + cpu(test) > budget and test not in shielded:
            continue
        if test not in dialect_reduced:
            spent += cpu(test)
        run.add(test)
    listed |= runnable - run
    return Triage(listed, run, held, shielded, bool(runnable - run))


def held_tests_to_list(held: set[int], job, failing_after_run: set[int], certain: bool, budget_cut: bool,
                       tests: list[dict], fixtures: FixtureIndex, many_failures: int) -> set[int]:
    """The held tests to list: all of them without usable verdicts, or when a recorded
    comparison flips; otherwise those with a failing run test of their class or fixture file
    (or all, past ``many_failures`` failures)."""
    if (job is None or job.killed or not job.verdicts
            or any(job.verdicts.get(t) is None for t in job.test_ids)):
        return set(held)
    position = {t["id"]: i for i, t in enumerate(tests)}
    failing = {position[test_id] for test_id, report in job.verdicts.items()
               if report["verdict"] in ("fail", "timeout") and test_id in position}
    failing |= {t for t in failing_after_run if tests[t]["id"] not in job.verdicts}  # class fixture errors
    if len(failing) + (1 if budget_cut else 0) >= many_failures:
        return set(held)
    failing_classes = {test_class(tests[t]["id"]) for t in failing}
    failing_fixtures = set().union(*(fixtures.of(t) for t in failing)) if failing else set()
    return {t for t in held if test_class(tests[t]["id"]) in failing_classes or failing_fixtures & fixtures.of(t)}
