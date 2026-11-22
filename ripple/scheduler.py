"""The parent side of a run: fork one child per commit and collect its verdicts.

Children run ``parallel`` at a time. Each gets a deadline from its tests' CPU limits; the whole
batch also stops two seconds before the global deadline. A killed or unfinished child leaves
some tests without a verdict, and the caller lists those.
"""

from __future__ import annotations

import json
import os
import select
import signal
import sys
import time
from dataclasses import dataclass, field

from .harness import ChildOrders, child_main
from .sandbox import fresh_mirror


@dataclass
class Job:
    commit: str
    orders: ChildOrders
    new_contents: dict
    verdicts: dict[str, dict] = field(default_factory=dict)  # test id -> its report
    class_errors: list[str] = field(default_factory=list)  # "file::Class" keys
    timings: dict[str, float] = field(default_factory=dict)  # "class_setup:..." / "module_import:..." -> CPU
    killed: bool = False
    pid: int = 0
    fd: int = -1
    pending: bytes = b""
    started_at: float = 0.0
    ready_at: float = 0.0
    give_up_at: float = 0.0

    @property
    def test_ids(self) -> list[str]:
        return self.orders.test_ids


def _kill(job: Job) -> None:
    job.killed = True
    try:
        os.kill(job.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _spawn(job: Job) -> None:
    if job.orders.swap is None:
        fresh_mirror(job.orders.tree, job.orders.pinned_root, job.new_contents)
    reader, writer = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        status = 0
        try:
            child_main(job.orders, writer)
        except BaseException as problem:  # noqa
            try:
                os.write(2, f"child for {job.commit} crashed: {problem!r}\n".encode())
            except Exception:
                pass
            status = 1
        os._exit(status)
    os.close(writer)
    job.pid, job.fd = pid, reader
    job.started_at = time.monotonic()
    limits = job.orders.cpu_limits
    job.give_up_at = job.started_at + 30.0 + 4.0 * sum(limits.get(t, 10.0) for t in job.test_ids)


def _absorb(job: Job, data: bytes) -> None:
    job.pending += data
    while b"\n" in job.pending:
        line, job.pending = job.pending.split(b"\n", 1)
        try:
            report = json.loads(line)
        except ValueError:
            continue
        kind = report["kind"]
        if kind == "verdict":
            job.verdicts[report["test"]] = report
        elif kind == "ready":
            job.ready_at = time.monotonic()
        elif kind == "class_error":
            job.class_errors.append(report["key"])
        else:
            job.timings[f"{kind}:{report['key']}"] = report["cpu"]


def _reap(job: Job, log) -> None:
    import shutil

    os.close(job.fd)
    try:
        os.waitpid(job.pid, 0)
    except ChildProcessError:
        pass
    if job.orders.swap is None:
        shutil.rmtree(job.orders.tree, ignore_errors=True)
    tally: dict[str, int] = {}
    for report in job.verdicts.values():
        tally[report["verdict"]] = tally.get(report["verdict"], 0) + 1
    wall = sum(r["wall"] for r in job.verdicts.values())
    cpu = sum(r["cpu"] for r in job.verdicts.values())
    slowest = sorted(job.verdicts.values(), key=lambda r: -r["wall"])[:3]
    setup = job.ready_at - job.started_at if job.ready_at else -1
    log(f"ran {job.commit}: {len(job.verdicts)}/{len(job.test_ids)} tests in {time.monotonic() - job.started_at:.1f}s "
        f"(wall {wall:.1f}s, cpu {cpu:.1f}s, {tally}, ready after {setup:.1f}s; slowest "
        + ", ".join(f"{r['test'].split('::')[-1]}={r['wall']:.2f}" for r in slowest) + ")"
        + (" KILLED" if job.killed else "") + (" swapped" if job.orders.swap is not None else ""))


def run_all(jobs: list[Job], parallel: int, deadline: float | None, log) -> None:
    import gc

    # The parent holds the atlas (millions of objects); every gc pass a test triggers in a child
    # would walk them all. Frozen objects are skipped by the collector.
    gc.collect()
    gc.freeze()
    waiting = list(jobs)
    running: dict[int, Job] = {}
    stop_at = None if deadline is None else deadline - 2.0
    while waiting or running:
        if stop_at is not None and time.monotonic() > stop_at:
            if waiting:
                log(f"deadline: {len(waiting)} commits not run")
            waiting.clear()
            for job in running.values():
                if not job.killed:
                    _kill(job)
        while waiting and len(running) < parallel:
            job = waiting.pop(0)
            _spawn(job)
            running[job.fd] = job
        if not running:
            break
        wake_at = min(job.give_up_at for job in running.values())
        if stop_at is not None:
            wake_at = min(wake_at, stop_at)  # a silent child must not hold the batch past it
        ready, _, _ = select.select(list(running), [], [], max(0.05, wake_at - time.monotonic()))
        for fd in ready:
            job = running[fd]
            data = os.read(fd, 65536)
            if data:
                _absorb(job, data)
            else:
                del running[fd]
                _reap(job, log)
        now = time.monotonic()
        for job in list(running.values()):
            if now > job.give_up_at:
                _kill(job)
