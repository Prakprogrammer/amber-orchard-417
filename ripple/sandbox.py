"""Doing risky things in a forked child: importing a patched tree, computing a table.

``in_child`` runs a function in a fork and returns its pickled result (None when the child
fails, hangs past its alarm, or dies). ``mirror_tree`` lays out a patched copy of the pinned
checkout made mostly of symlinks, so building one costs milliseconds.
"""

from __future__ import annotations

import os
import pickle
import shutil
import signal
import sys


def in_child(work, time_limit_s: float, silence_stderr: bool = False):
    reader, writer = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(reader)
            sink = os.open(os.devnull, os.O_WRONLY)
            os.dup2(sink, 1)
            if silence_stderr:
                os.dup2(sink, 2)
            signal.alarm(int(time_limit_s))
            payload = pickle.dumps(work())
        except BaseException:  # noqa: whatever happened, the parent gets "no answer"
            payload = pickle.dumps(None)
        view = memoryview(payload)
        while view:
            view = view[os.write(writer, view):]
        os._exit(0)
    os.close(writer)
    chunks = []
    while chunk := os.read(reader, 1 << 16):
        chunks.append(chunk)
    os.close(reader)
    os.waitpid(pid, 0)
    try:
        return pickle.loads(b"".join(chunks))
    except Exception:  # noqa: a child killed mid-write
        return None


def mirror_tree(destination: str, pinned_root: str, new_contents: dict[str, str | None]) -> None:
    """``destination`` mirrors ``pinned_root`` through symlinks, except for the patched files,
    which are written out (a None content leaves the file out)."""
    layout: dict = {}
    for path, text in new_contents.items():
        folder = layout
        *directories, file_name = path.split("/")
        for directory in directories:
            if not isinstance(folder.get(directory), dict):
                folder[directory] = {}
            folder = folder[directory]
        folder[file_name] = ("file", text)

    def build(source_dir: str | None, target_dir: str, entries: dict) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if source_dir and os.path.isdir(source_dir):
            for name in os.listdir(source_dir):
                if name not in entries:
                    os.symlink(os.path.join(source_dir, name), os.path.join(target_dir, name))
        for name, entry in entries.items():
            if isinstance(entry, tuple):
                if entry[1] is not None:
                    with open(os.path.join(target_dir, name), "w", encoding="utf-8", newline="") as handle:
                        handle.write(entry[1])
            else:
                build(os.path.join(source_dir, name) if source_dir else None, os.path.join(target_dir, name), entry)

    build(pinned_root, destination, layout)


def enter_tree(root: str, pinned_root: str) -> None:
    """Make ``root`` the tree this process imports sqlglot and the tests from."""
    pinned = os.path.realpath(pinned_root)
    sys.path[:] = [root] + [p for p in sys.path if os.path.realpath(p) != pinned and p not in ("", ".")]
    os.chdir(root)


def fresh_mirror(destination: str, pinned_root: str, new_contents: dict[str, str | None]) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    mirror_tree(destination, pinned_root, new_contents)


class ImportProbe:
    """Which modules (all of sqlglot, plus the test modules) fail to import with a patch
    applied; None when the probe itself fails or hangs. Results are kept per patch."""

    def __init__(self, pinned_root: str, test_modules: list[str], scratch: str, time_limit_s: float):
        self.pinned_root = pinned_root
        self.test_modules = test_modules
        self.mirror = os.path.join(scratch, "import-probe")
        self.time_limit_s = time_limit_s
        self._known: dict[tuple, set | None] = {}

    def failures(self, new_contents: dict[str, str | None]) -> set[str] | None:
        key = tuple(sorted((path, hash(text)) for path, text in new_contents.items()))
        if key not in self._known:
            self._known[key] = in_child(lambda: self._import_everything(new_contents), self.time_limit_s,
                                        silence_stderr=True)
        return self._known[key]

    def breaks_imports(self, new_contents: dict[str, str | None]) -> bool:
        """Does the patched tree fail (or hang) on an import that the pinned tree passes?"""
        broken = self.failures(new_contents)
        if broken is None:
            return True
        if not broken:
            return False
        pinned = self.failures({})
        return pinned is None or bool(broken - pinned)

    def _import_everything(self, new_contents) -> set[str]:
        import importlib
        import pkgutil

        from .harness import run_pools_inline

        fresh_mirror(self.mirror, self.pinned_root, new_contents)
        enter_tree(self.mirror, self.pinned_root)
        os.environ.setdefault("SKIP_INTEGRATION", "0")
        run_pools_inline()
        failed = set()
        try:
            import sqlglot

            names = [info.name for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.")]
        except BaseException:  # noqa
            names = []
            failed.add("sqlglot")
        for name in names + self.test_modules:
            try:
                importlib.import_module(name)
            except BaseException:  # noqa
                failed.add(name)
        return failed
