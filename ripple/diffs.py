"""Reading and applying git-style unified diffs, with no external tools.

A diff is read into one ``FileDiff`` per touched file. ``FileDiff.source`` is the path before
the change (``None`` for an added file) and ``FileDiff.target`` the path after it (``None`` for
a deleted file). Every hunk row keeps its one-character marker: ``" "`` context, ``"-"``
removed, ``"+"`` added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_HEADER = re.compile(r'^diff --git ("?a/.+?"?) ("?b/.+"?)$')
_NO_EOL = "\\ No newline at end of file"


class DiffDoesNotApply(Exception):
    """A hunk's old lines were not found in the file."""


@dataclass
class Hunk:
    old_first: int
    old_count: int
    new_first: int
    new_count: int
    rows: list[str] = field(default_factory=list)
    old_lacks_eol: bool = False
    new_lacks_eol: bool = False

    def side(self, markers: str) -> list[str]:
        return [row[1:] for row in self.rows if row[:1] in markers]


@dataclass
class FileDiff:
    source: str | None
    target: str | None
    hunks: list[Hunk] = field(default_factory=list)
    is_binary: bool = False

    @property
    def name(self) -> str:
        return self.target or self.source or ""

    def rows_marked(self, marker: str) -> list[str]:
        return [row[1:] for hunk in self.hunks for row in hunk.rows if row[:1] == marker]

    @property
    def removed_count(self) -> int:
        return sum(1 for hunk in self.hunks for row in hunk.rows if row[:1] == "-")

    @property
    def added_count(self) -> int:
        return sum(1 for hunk in self.hunks for row in hunk.rows if row[:1] == "+")


def _clean_path(raw: str) -> str | None:
    path = raw.strip().split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path[:2] in ("a/", "b/"):
        return path[2:]
    return path


class _DiffReader:
    def __init__(self) -> None:
        self.files: list[FileDiff] = []
        self.current: FileDiff | None = None
        self.hunk: Hunk | None = None
        self.old_left = 0
        self.new_left = 0
        self.last_marker = " "

    def feed(self, line: str) -> None:
        if self.hunk is not None and (self.old_left > 0 or self.new_left > 0):
            self._hunk_row(line)
        else:
            self._header_line(line)

    def _hunk_row(self, line: str) -> None:
        if line.startswith("\\"):
            return
        marker, body = (line[:1], line[1:]) if line else (" ", "")
        if marker == "-":
            self.old_left -= 1
        elif marker == "+":
            self.new_left -= 1
        else:  # context, or a malformed row read as context
            marker = " "
            self.old_left -= 1
            self.new_left -= 1
        self.hunk.rows.append(marker + body)
        self.last_marker = marker

    def _header_line(self, line: str) -> None:
        if line.startswith(_NO_EOL):
            if self.hunk is not None:
                if self.last_marker != "+":
                    self.hunk.old_lacks_eol = True
                if self.last_marker != "-":
                    self.hunk.new_lacks_eol = True
            return
        if line.startswith("diff --git "):
            found = _GIT_HEADER.match(line)
            paths = (_clean_path(found.group(1)), _clean_path(found.group(2))) if found else (None, None)
            self.current = FileDiff(*paths)
            self.files.append(self.current)
            self.hunk = None
            return
        if line.startswith("--- "):
            if self.current is None:
                self.current = FileDiff(None, None)
                self.files.append(self.current)
            self.current.source = _clean_path(line[4:])
            return
        if line.startswith("+++ "):
            if self.current is not None:
                self.current.target = _clean_path(line[4:])
            return
        if self.current is not None:
            if line.startswith("new file mode"):
                self.current.source = None
            elif line.startswith("deleted file mode"):
                self.current.target = None
            elif line.startswith("rename from "):
                self.current.source = line[len("rename from "):]
            elif line.startswith("rename to "):
                self.current.target = line[len("rename to "):]
            elif line.startswith(("Binary files", "GIT binary patch")):
                self.current.is_binary = True
        header = _HUNK_HEADER.match(line)
        if header and self.current is not None:
            old_first, old_count, new_first, new_count = header.groups()
            self.hunk = Hunk(int(old_first), 1 if old_count is None else int(old_count),
                             int(new_first), 1 if new_count is None else int(new_count))
            self.current.hunks.append(self.hunk)
            self.old_left, self.new_left = self.hunk.old_count, self.hunk.new_count


def read_diff(text: str) -> list[FileDiff]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    reader = _DiffReader()
    for line in lines:
        reader.feed(line)
    return reader.files


def _locate(lines: list[str], wanted: list[str], hint: int, floor: int) -> int | None:
    """Where ``wanted`` occurs in ``lines``: at ``hint`` if possible, else the nearest
    position (trying below before above at each distance) that is not before ``floor``."""
    if lines[hint:hint + len(wanted)] == wanted:
        return hint
    last = len(lines) - len(wanted)
    for distance in range(1, len(lines) + 1):
        for spot in (hint - distance, hint + distance):
            if floor <= spot <= last and lines[spot:spot + len(wanted)] == wanted:
                return spot
    return None


def patch_text(old_text: str | None, diff: FileDiff) -> str | None:
    """The file after the diff (``None`` when the diff deletes it)."""
    if diff.target is None:
        return None
    old_text = old_text or ""
    had_final_newline = old_text.endswith("\n")
    old_lines = old_text.split("\n") if old_text else []
    if had_final_newline:
        old_lines.pop()
    ends_with_newline = had_final_newline if old_lines else True
    result: list[str] = []
    cursor = 0
    for hunk in diff.hunks:
        before, after = hunk.side(" -"), hunk.side(" +")
        hint = max(hunk.old_first - 1 if hunk.old_count > 0 else hunk.old_first, 0)
        at = _locate(old_lines, before, hint, cursor)
        if at is None:
            raise DiffDoesNotApply(f"hunk @@ -{hunk.old_first} does not apply to {diff.name}")
        result += old_lines[cursor:at]
        result += after
        cursor = at + len(before)
        if cursor >= len(old_lines):
            if hunk.new_lacks_eol:
                ends_with_newline = False
            elif hunk.old_lacks_eol or not had_final_newline:
                ends_with_newline = True
    result += old_lines[cursor:]
    text = "\n".join(result)
    return text + "\n" if result and ends_with_newline else text


def touched_old_lines(diff: FileDiff) -> tuple[set[int], set[int]]:
    """(old line numbers removed or rewritten, old line numbers bordering a pure insertion)."""
    rewritten: set[int] = set()
    borders: set[int] = set()
    for hunk in diff.hunks:
        line = hunk.old_first if hunk.old_count > 0 else hunk.old_first + 1
        inserting = False
        for row in hunk.rows:
            marker = row[:1]
            if marker == "+":
                if not inserting:
                    borders.add(line - 1)
                inserting = True
                continue
            if marker == " " and inserting:
                borders.add(line)
            elif marker == "-":
                rewritten.add(line)
            inserting = False
            line += 1
        if inserting:
            borders.add(line)
    return rewritten, borders


def added_new_lines(diff: FileDiff) -> set[int]:
    """New-file line numbers of the added rows."""
    added: set[int] = set()
    for hunk in diff.hunks:
        line = hunk.new_first if hunk.new_count > 0 else hunk.new_first + 1
        for row in hunk.rows:
            marker = row[:1]
            if marker == "+":
                added.add(line)
            if marker in " +":
                line += 1
    return added
