"""Every tunable, read once from the environment (``RIPPLE_*``)."""

from __future__ import annotations

import os
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.dirname(_HERE)


def _text(name: str, default: str) -> str:
    return os.environ.get(f"RIPPLE_{name}", default)


def _number(name: str, default: float) -> float:
    return float(_text(name, str(default)))


def _switch(name: str, default: bool) -> bool:
    return _text(name, "1" if default else "0") == "1"


@dataclass(frozen=True)
class Settings:
    data_dir: str
    sqlglot_root: str
    coverage_atlas: str
    comparison_atlas: str
    pass_atlas: str
    tpcds_atlas: str
    cost_table: str
    dialect_families: str
    tpc_rows: str
    scratch: str

    list_only: bool  # never run anything here: list every candidate
    use_comparisons: bool
    one_cpu: bool
    parallel: int
    seconds_per_commit: float  # the time we get, per commit, for the whole run

    held_above_s: float  # tests longer than this are held and listed on evidence only
    held_run_share: float  # ...unless running them costs at most this share of their seconds
    always_run_below_s: float  # a test whose run costs less than this always runs
    run_ratio: float  # run a test when its run costs at most this times its listing
    dialect_run_ratio: float  # the same, for dialect-reduced fixture tests
    dialect_base_s: float  # fixed cost of a dialect-reduced test's method around its loop
    swap_commit_budget_s: float  # CPU a hot-swapped commit's runs may use
    fresh_commit_budget_s: float  # CPU a commit needing a fresh import may use
    budget_slack: float  # planned costs run about 2x the CPU really used
    after_run_s: float  # expected cost of listing what a run finds failing
    after_certain_run_s: float  # the same, when a recorded comparison flips
    many_failures: int  # this many failures list the held tests too

    import_guard: bool  # check that a patch editing import-time code still imports
    import_check_limit_s: float
    import_check_window: float  # import checks only in the first part of the time
    fixture_reduction: bool
    dialect_reduction: bool
    tpcds_stand_in: bool
    tpcds_stand_in_cpu: float  # planned CPU of the stand-in when it checks edited lines
    tpcds_full_check_cpu: float  # planned CPU when it checks every query

    @classmethod
    def from_environment(cls) -> "Settings":
        atlas = _text("ATLAS", os.path.join(_APP, "atlas"))
        cache = _text("CACHE", os.path.join(_APP, "cache"))
        one_cpu = _switch("ONE_CPU", True)
        return cls(
            data_dir=_text("DATA", "/data"),
            sqlglot_root=_text("SQLGLOT", "/opt/sqlglot"),
            coverage_atlas=os.path.join(atlas, "coverage.pkl.gz"),
            comparison_atlas=os.path.join(atlas, "comparisons.pkl.gz"),
            pass_atlas=os.path.join(atlas, "passes.pkl.gz"),
            tpcds_atlas=os.path.join(atlas, "tpcds.pkl.gz"),
            cost_table=os.path.join(atlas, "costs.json"),
            dialect_families=os.path.join(cache, "dialect_families.pkl"),
            tpc_rows=os.path.join(cache, "tpc_rows.pkl"),
            scratch=_text("SCRATCH", "/tmp/ripple"),
            list_only=_switch("LIST_ONLY", False),
            use_comparisons=_switch("COMPARISONS", True),
            one_cpu=one_cpu,
            parallel=int(_number("PARALLEL", 1 if one_cpu else 2)),
            seconds_per_commit=_number("SECONDS_PER_COMMIT", 3.4),
            held_above_s=_number("HELD_ABOVE_S", 5.0),
            held_run_share=_number("HELD_RUN_SHARE", 0.6),
            always_run_below_s=_number("ALWAYS_RUN_BELOW_S", 0.5),
            run_ratio=_number("RUN_RATIO", 1.0),
            dialect_run_ratio=_number("DIALECT_RUN_RATIO", 2.0),
            dialect_base_s=_number("DIALECT_BASE_S", 1.0),
            swap_commit_budget_s=_number("SWAP_COMMIT_BUDGET_S", 8.0),
            fresh_commit_budget_s=_number("FRESH_COMMIT_BUDGET_S", 5.0),
            budget_slack=_number("BUDGET_SLACK", 2.5),
            after_run_s=_number("AFTER_RUN_S", 1.0),
            after_certain_run_s=_number("AFTER_CERTAIN_RUN_S", 1.9),
            many_failures=int(_number("MANY_FAILURES", 100000)),
            import_guard=_switch("IMPORT_GUARD", True),
            import_check_limit_s=_number("IMPORT_CHECK_LIMIT_S", 10),
            import_check_window=_number("IMPORT_CHECK_WINDOW", 0.3),
            fixture_reduction=_switch("FIXTURE_REDUCTION", True),
            dialect_reduction=_switch("DIALECT_REDUCTION", True),
            tpcds_stand_in=_switch("TPCDS_STAND_IN", True),
            tpcds_stand_in_cpu=_number("TPCDS_STAND_IN_CPU", 6.0),
            tpcds_full_check_cpu=_number("TPCDS_FULL_CHECK_CPU", 1.0),
        )
