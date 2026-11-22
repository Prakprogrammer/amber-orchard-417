"""The whole run: read the commits, plan each one, run what pays off, print the worker lists.

Stages, per commit (``Pipeline.plan``):
  1. impact analysis -> candidate tests (and the data readers among them);
  2. an import probe when an edited function body runs during import;
  3. reductions: pass subsets, edited fixture entries, dialect families, miniature variants and
     the TPC-DS stage check;
  4. triage: run, list or hold each candidate.
Then, for all commits together: keep the planned CPU inside the time available, run the
chosen tests in forked children, and turn verdicts and evidence into the final lists.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

from . import bits
from .atlas import CoverageAtlas, load_costs, load_passes, read_artifact, test_module
from .costs import CostModel, RunPricing, listing_seconds, pack
from .diffs import FileDiff, read_diff
from .impact import Impact, ImpactAnalyzer
from .reductions import DialectFamilies, Reductions, edited_statements, narrow_fixture_readers, narrow_to_dialects
from .settings import Settings
from .standins import TPCDS_TEST, Miniature, OnlyPasses, TpcdsStageCheck, pass_kind
from .syntax import edited_functions
from .triage import FixtureIndex, Triage, held_tests_to_list, triage

STARTED = time.monotonic()
MINIATURE_CPU = {  # our CPU seconds for the small-scale variants, measured on the pinned tree
    "tests/test_parser.py::TestParser::test_max_nodes": 0.4,
    "tests/test_optimizer.py::TestOptimizer::test_scope": 0.5,
    "tests/test_generator.py::TestGenerator::test_generator_memory": 5.5,  # may load every dialect
}
TPCDS_INDEPENDENT_OF = {"sqlglot/executor/__init__.py", "sqlglot/serde.py"}  # the stage check can't vouch for these
_HEAVY_TEST_MODULES = ("tests/test_optimizer.py", "tests/test_executor.py")


def log(*parts) -> None:
    print(f"[{time.monotonic() - STARTED:6.2f}]", *parts, file=sys.stderr, flush=True)


def only_sqlglot_sources(new_contents: dict) -> bool:
    return bool(new_contents) and all(text is not None and path.startswith("sqlglot/") and path.endswith(".py")
                                      for path, text in new_contents.items())


def edits_outside_functions(impact: Impact | None) -> bool:
    """Does any .py edit lie outside function bodies (class/module level, added, removed or
    renamed files, cosmetic-only files)?"""
    if impact is None or impact.everything:
        return True
    for path in impact.new_contents:
        if not path.endswith(".py"):
            continue
        edited = impact.trees.get(path)
        if edited is None or edited.old is None or edited.new is None:
            return True
        if edited_functions(edited.old, edited.new, edited.removed, edited.added) is None:
            return True
    return False


@dataclass
class CommitPlan:
    commit: str
    impact: Impact | None
    candidates: set[int]
    data_readers: set[int]
    pricing: RunPricing
    reductions: Reductions
    plan: Triage
    hot_swap: bool = False
    import_time_edit: bool = False  # an edited function body runs during import (and imports still work)
    tpcds_last: int | None = None
    list_everything_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def certain(self) -> bool:
        return bool(self.impact is not None and self.impact.certain)


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

    # ---- loading ----------------------------------------------------------------------
    def load(self) -> None:
        settings = self.settings
        with open(os.path.join(settings.data_dir, "tests.json")) as handle:
            self.tests = json.load(handle)["tests"]
        self.ids = [t["id"] for t in self.tests]
        self.position = {test_id: i for i, test_id in enumerate(self.ids)}
        self.durations = [round(t["seconds"] * 1000) for t in self.tests]
        commits_dir = os.path.join(settings.data_dir, "commits")
        self.commits = sorted(name[:-len(".patch")] for name in os.listdir(commits_dir) if name.endswith(".patch"))
        self.commits_dir = commits_dir
        self.deadline = STARTED + max(20.0, settings.seconds_per_commit * len(self.commits))
        self.atlas = CoverageAtlas(read_artifact(settings.coverage_atlas), self.ids)
        comparisons = None
        if settings.use_comparisons and os.path.exists(settings.comparison_atlas):
            try:
                from .compare import ComparisonAtlas

                comparisons = ComparisonAtlas(read_artifact(settings.comparison_atlas), self.ids)
            except Exception as problem:  # noqa: recorded for another test list: line coverage only
                log(f"comparison atlas not used: {problem!r}")
        self.analyzer = ImpactAnalyzer(self.atlas, self.tests, settings.sqlglot_root, comparisons)
        self.model = CostModel(self.tests, load_costs(settings.cost_table))
        self.passes = {}
        if not settings.list_only and os.path.exists(settings.pass_atlas):
            self.passes = load_passes(settings.pass_atlas, self.ids)
        self.fixtures = FixtureIndex(self.atlas)
        self.families = DialectFamilies(settings.dialect_families, settings.sqlglot_root, log)
        from .sandbox import ImportProbe

        test_modules = sorted({t["file"][:-3].replace("/", ".") for t in self.tests})
        self.import_probe = ImportProbe(settings.sqlglot_root, test_modules, settings.scratch,
                                        settings.import_check_limit_s)
        log(f"{len(self.tests)} tests, {len(self.commits)} commits; {self.deadline - STARTED:.0f}s available")

    # ---- planning one commit ----------------------------------------------------------
    def plan(self, commit: str) -> CommitPlan:
        settings = self.settings
        with open(os.path.join(self.commits_dir, commit + ".patch"), encoding="utf-8", errors="replace", newline="") as handle:
            patch = handle.read()
        diffs = read_diff(patch)
        try:
            impact = self.analyzer.analyze(diffs)
        except Exception as problem:  # never crash: every test is a candidate
            log(f"{commit}: impact analysis crashed {problem!r}")
            impact = None
        decided = impact is not None and impact.decided
        everything = impact is None or impact.everything
        import_broken = decided and impact.reaches_import_time and settings.import_guard and self._import_broken(impact)
        notes: list[str] = []
        if import_broken:
            candidates, data_readers = set(range(len(self.tests))), set()
            notes.append("EVERYTHING (the patched tree fails to import)")
        elif everything:
            candidates, data_readers = set(range(len(self.tests))), set()
            notes.append("EVERYTHING")
        elif impact.unchanged:
            candidates, data_readers = set(), set()
            notes.append("nothing changed")
        else:
            candidates = set(bits.members(impact.candidates | self.atlas.unrecorded))
            data_readers = set(bits.members(impact.data_readers))
            notes.append(f"{len(candidates)} candidates")

        reductions = Reductions()
        pricing = RunPricing(fresh_import=impact is None or impact.python_changed, reduced=reductions.reduced_cpu,
                             after_run_s=settings.after_certain_run_s if impact is not None and impact.certain
                             else settings.after_run_s)
        hot_swap = False
        may_run = not settings.list_only and impact is not None and not import_broken
        if may_run and candidates and only_sqlglot_sources(impact.new_contents):
            from .hotswap import could_swap

            try:
                hot_swap = could_swap(impact.new_contents, self.atlas.import_time_lines, impact.trees)
            except Exception:  # noqa
                hot_swap = False
        pricing.hot_swap = hot_swap
        if hot_swap and self.passes:  # (also when another file of the patch made it "everything")
            self._keep_reaching_passes(diffs, candidates, reductions, notes)
        if may_run and not everything:
            self._reduce(impact, diffs, patch, candidates, data_readers, hot_swap, reductions, notes)
        if may_run and not everything and candidates:
            plan = triage(candidates, data_readers, hot_swap, pricing, reductions.dialect_reduced, self.model,
                          self.tests, self.fixtures, settings)
        else:
            plan = Triage(set(candidates), set(), set(), set(), False)
        result = CommitPlan(commit, impact, candidates, data_readers, pricing, reductions, plan, hot_swap,
                            import_time_edit=decided and impact.reaches_import_time and not import_broken,
                            list_everything_s=listing_seconds(candidates, self.durations), notes=notes)
        self._place_tpcds_last(result)
        reductions.stand_ins = {t: d for t, d in reductions.stand_ins.items() if self.position[t] in plan.run}
        log(f"{commit}: {'; '.join(notes)} list={len(plan.listed)} run={len(plan.run)} held={len(plan.held)}"
            f"{' swap' if hot_swap else ''} | " + ("; ".join(impact.notes)[:400] if impact else "crashed"))
        return result

    def _import_broken(self, impact: Impact) -> bool:
        """A changed statement runs while sqlglot is imported: the traced consumers only hold
        while the import still completes. Late in the run there is no time to check (each
        check imports everything, and a hang waits for its limit): assume it breaks."""
        elapsed = time.monotonic() - STARTED
        if elapsed > self.settings.import_check_window * (self.deadline - STARTED):
            return True
        return self.import_probe.breaks_imports(impact.new_contents)

    def _keep_reaching_passes(self, diffs: list[FileDiff], candidates: set[int], reductions: Reductions,
                              notes: list[str]) -> None:
        """Looping tests run only the passes reaching an edited line; none -> not a candidate."""
        edited = edited_statements(self.analyzer, diffs)
        run_cpu = self.model.run_cpu
        for test_id, test_passes in self.passes.items():
            test = self.position[test_id]
            if test not in candidates or edited is None:
                continue
            kept = test_passes.reaching(edited)
            if not kept:
                candidates.discard(test)
                notes.append(f"{test_id.split('::')[-1]} reaches no edit")
                continue
            reductions.stand_ins[test_id] = OnlyPasses(frozenset(kept), pass_kind(test_passes.labels))
            reductions.reduced_cpu[test] = run_cpu[test] * len(kept) / len(test_passes.labels)

    def _reduce(self, impact: Impact, diffs: list[FileDiff], patch: str, candidates: set[int], data_readers: set[int],
                hot_swap: bool, reductions: Reductions, notes: list[str]) -> None:
        settings, run_cpu = self.settings, self.model.run_cpu
        if data_readers and settings.fixture_reduction:
            settled = narrow_fixture_readers(self.analyzer, impact, self.tests, data_readers, candidates, self.passes,
                                             run_cpu, reductions)
            if settled:
                data_readers -= settled
                notes.append(f"fixture entries settle {len(settled)} readers")
        if (settings.dialect_reduction and not hot_swap and not impact.unchanged and self.passes
                and edits_outside_functions(impact)):
            count = narrow_to_dialects(self.families, self.analyzer, impact, patch, self.position, candidates,
                                       data_readers, self.passes, run_cpu, settings.dialect_base_s, reductions)
            if count:
                notes.append(f"dialect entries reduce {count} tests")
        for test_id, cpu in MINIATURE_CPU.items():  # tests that only check behaviour at a size
            test = self.position.get(test_id)
            if test is None or test not in candidates or test in data_readers or test_id in reductions.stand_ins:
                continue
            if test_id.split("::")[0] in impact.new_contents:
                continue  # the test file itself changes: the pinned text would not match
            reductions.stand_ins[test_id] = Miniature()
            reductions.reduced_cpu[test] = cpu
        if settings.tpcds_stand_in:
            self._tpcds_stand_in(impact, diffs, candidates, data_readers, hot_swap, reductions, notes)

    def _tpcds_stand_in(self, impact, diffs, candidates, data_readers, hot_swap, reductions, notes) -> None:
        test = self.position.get(TPCDS_TEST)
        if (test is None or test not in candidates or test in data_readers or not os.path.exists(self.settings.tpcds_atlas)
                or any(p in TPCDS_INDEPENDENT_OF or p.startswith("tests/") for p in impact.new_contents)):
            return
        edited = edited_statements(self.analyzer, diffs) if hot_swap else None
        if edited is None and all(p.endswith(".py") for p in impact.new_contents):
            # namespace-level edits, or lines unclear: every executed query runs patched
            reductions.stand_ins[TPCDS_TEST] = TpcdsStageCheck(None, None)
            notes.append("tpcds full stage check")
        elif edited is not None:
            test_passes = self.passes.get(TPCDS_TEST)
            queries = None if test_passes is None else frozenset(label[1] for label in test_passes.reaching(edited))
            reductions.stand_ins[TPCDS_TEST] = TpcdsStageCheck(edited, queries)
            notes.append("tpcds stage check")

    def _place_tpcds_last(self, plan: CommitPlan) -> None:
        """The TPC-DS stand-in runs after the commit's other tests: it decides from their
        verdicts first (cheaply), and only then checks stages."""
        test = self.position.get(TPCDS_TEST)
        directive = plan.reductions.stand_ins.get(TPCDS_TEST)
        if isinstance(directive, TpcdsStageCheck) and test in plan.plan.run:
            plan.tpcds_last = test
        elif directive is not None and test in plan.plan.held and plan.plan.run:
            plan.plan.held.discard(test)
            plan.plan.run.add(test)
            plan.tpcds_last = test
            full_check = isinstance(directive, TpcdsStageCheck) and directive.edited_lines is None
            plan.reductions.reduced_cpu[test] = (self.settings.tpcds_full_check_cpu if full_check
                                                 else self.settings.tpcds_stand_in_cpu)
        elif isinstance(directive, TpcdsStageCheck):
            del plan.reductions.stand_ins[TPCDS_TEST]

    # ---- all commits together ---------------------------------------------------------
    def fit_budget(self, plans: list[CommitPlan]) -> None:
        """Keep the planned CPU inside the time left: drop the runs saving least per second."""
        def run_cpu(plan: CommitPlan) -> float:
            return self.model.run_cost(plan.plan.run, plan.pricing)

        budget = self.settings.parallel * (self.deadline - time.monotonic() - 3.0) * 0.8 * self.settings.budget_slack
        planned = sum(run_cpu(p) for p in plans)
        if planned <= budget:
            return
        for plan in sorted((p for p in plans if p.plan.run), key=lambda p: p.list_everything_s / max(run_cpu(p), 1e-3)):
            if planned <= budget:
                break
            planned -= run_cpu(plan)
            plan.plan.listed = plan.candidates | plan.plan.held
            plan.plan.run, plan.plan.held = set(), set()
        log(f"over budget: runs trimmed to {planned:.0f}s of CPU (budget {budget:.0f}s)")

    def execute(self, plans: list[CommitPlan]) -> dict:
        from .harness import ChildOrders, run_pools_inline
        from .hotswap import Preload
        from .scheduler import Job, run_all
        from .tpc_rows import usable

        settings = self.settings
        needed = [p for p in plans if p.plan.run]
        if not needed:
            return {}
        heavy = any(self.ids[t].startswith(_HEAVY_TEST_MODULES) for p in needed for t in p.plan.run)
        _preload_libraries(heavy, settings.sqlglot_root)
        # what the swapping children reuse: sqlglot, and the test modules they run
        modules = ["sqlglot"]
        for p in needed:
            if only_sqlglot_sources(p.impact.new_contents):
                modules += sorted({test_module(self.ids[t]) for t in p.plan.run})
        run_pools_inline()  # test modules bind ProcessPoolExecutor when imported
        preload = Preload(settings.sqlglot_root)
        preload.import_modules(list(dict.fromkeys(modules)), log)
        log(f"preloaded {len(modules)} modules; {len(preload.ran)} sqlglot functions ran meanwhile")
        jobs = []
        for p in sorted(needed, key=lambda p: self.model.run_cost(p.plan.run, p.pricing)):
            order = sorted(p.plan.run, key=lambda t: (t == p.tpcds_last, t not in p.plan.shielded, t))
            # CPU guard: a test may use about what listing it would cost; then it is stopped and listed
            limits = {self.ids[t]: max(1.0, 3.0 * self.model.test_cpu(t, p.pricing),
                                       self.tests[t]["seconds"] if self.tests[t]["seconds"] >= 2.0 else 0.0)
                      for t in p.plan.run}
            orders = ChildOrders(
                tree=os.path.join(settings.scratch, "jobs", p.commit), pinned_root=settings.sqlglot_root,
                test_ids=[self.ids[t] for t in order], cpu_limits=limits,
                reimport_sqlglot=p.impact.sqlglot_python_changed, stand_ins=p.reductions.stand_ins,
                use_tpc_rows=usable(settings.tpc_rows, p.impact.new_contents), tpc_rows_path=settings.tpc_rows)
            if only_sqlglot_sources(p.impact.new_contents):
                try:
                    orders.swap = preload.plan(p.impact.new_contents)
                except Exception as problem:  # noqa
                    log(f"{p.commit}: swap planning failed {problem!r}")
            jobs.append(Job(p.commit, orders, p.impact.new_contents))
        log(f"running {len(jobs)} commits ({sum(j.orders.swap is not None for j in jobs)} by swapping code), "
            f"~{sum(self.model.run_cost(p.plan.run, p.pricing) for p in needed):.0f}s CPU planned")
        run_all(jobs, settings.parallel, self.deadline, log)
        return {job.commit: job for job in jobs}

    def final_list(self, plan: CommitPlan, job) -> set[int]:
        listed = set(plan.plan.listed)
        if job is not None:
            for test_id in job.test_ids:
                report = job.verdicts.get(test_id)
                if report is None or report["verdict"] != "pass":
                    listed.add(self.position[test_id])
            for klass in job.class_errors:
                # a failed class fixture: in a full run its last test carries the teardown error
                members = [i for i, test_id in enumerate(self.ids) if test_id.startswith(klass + "::")]
                if members:
                    listed.add(members[-1])
            failing = sorted(listed - plan.plan.listed)
            if failing:
                log(f"{plan.commit}: failing after run: {[self.ids[t].split('::')[-1] for t in failing][:12]}")
        if plan.plan.held:
            listed |= held_tests_to_list(plan.plan.held, job, listed - plan.plan.listed, plan.certain,
                                         plan.plan.budget_cut, self.tests, self.fixtures, self.settings.many_failures)
        if plan.import_time_edit and self.settings.import_guard:
            # import-time code changed: a failure (or no run) means the import-time state changed
            # in a way that may reach tests the trace did not attribute to it
            passed = set()
            if job is not None:
                passed = {self.position[t] for t, r in job.verdicts.items() if r["verdict"] == "pass" and t in self.position}
            if job is None or job.killed or not passed:
                log(f"{plan.commit}: import-time edit with failures: every test not seen passing")
                listed |= set(range(len(self.tests))) - passed
        return listed

    def run(self) -> list[str]:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        if self.settings.one_cpu and not self.settings.list_only:
            _pin_to_one_cpu()
        self.load()
        plans = [self.plan(commit) for commit in self.commits]
        self.fit_budget(plans)
        jobs = {} if self.settings.list_only else self.execute(plans)
        lines, total_ms = [], 0
        for plan in plans:
            packing = pack(self.final_list(plan, jobs.get(plan.commit)), self.durations)
            total_ms += packing.cost_ms
            lines.append(json.dumps({"commit": plan.commit, "workers": packing.workers}))
        log(f"done; listed work {total_ms / 1000:.1f}s if nothing was missed{_container_cpu()}")
        return lines


def _pin_to_one_cpu() -> None:
    """Children run one after another: on one CPU, idle numpy / duckdb threads can't burn
    charged CPU on other cores."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) > 1:
            os.sched_setaffinity(0, {cpus[0]})
    except (AttributeError, OSError):
        pass


def _preload_libraries(with_dataframes: bool, sqlglot_root: str) -> None:
    began = time.monotonic()
    import concurrent.futures  # noqa: F401
    import unittest  # noqa: F401
    import unittest.mock  # noqa: F401

    if with_dataframes:  # test_optimizer / test_executor import these at module level
        for name in ("duckdb", "pandas", "pandas.testing"):
            try:
                __import__(name)
            except Exception:  # noqa
                pass
    sys.path.insert(0, sqlglot_root)
    log(f"libraries preloaded in {time.monotonic() - began:.2f}s")


def _container_cpu() -> str:
    try:
        with open("/sys/fs/cgroup/cpu.stat") as handle:
            for line in handle:
                if line.startswith("usage_usec"):
                    return f"; container CPU {int(line.split()[1]) / 1e6:.1f}s"
    except OSError:
        pass
    return ""


def list_everything(settings: Settings) -> list[str]:
    """The last resort when anything failed before printing: every test for every commit
    (never a miss)."""
    with open(os.path.join(settings.data_dir, "tests.json")) as handle:
        tests = json.load(handle)["tests"]
    workers = pack(range(len(tests)), [round(t["seconds"] * 1000) for t in tests]).workers
    commits = sorted(n[:-len(".patch")] for n in os.listdir(os.path.join(settings.data_dir, "commits")) if n.endswith(".patch"))
    return [json.dumps({"commit": commit, "workers": workers}) for commit in commits]
