"""Synthetic single-change patches against the pinned sqlglot, for measuring selection recall.

    python -m forge.mutants general <sqlglot checkout> <out commits dir> <count> [seed] [kind]
    python -m forge.mutants comparisons <sqlglot checkout> <tests.json> <out dir> [count] [seed]

``general``: edits on executed lines (operator swaps, flipped conditions, string tweaks, deleted
statements), class-level table entries and constants, and .sql fixtures; ``kind=import`` makes
only edits of function lines that run during import. ``comparisons``: operator swaps and
literal changes of recorded comparison sites, plus cands.json (each patch's candidates) for
``forge.labels``.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import random
import re
import sys

from ripple.atlas import read_artifact

ATLAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atlas")
SWAPS = [(" == ", " != "), (" != ", " == "), (" is not ", " is "), (" is ", " is not "), (" not in ", " in "),
         (" in ", " not in "), (" and ", " or "), (" or ", " and "), (" < ", " <= "), (" <= ", " < "), (" > ", " >= "),
         (" >= ", " > "), ("True", "False"), ("False", "True"), (r" \+ 1", " + 2"), (" - 1", " - 2"),
         (r"\[0\]", "[-1]"), (r"\[-1\]", "[0]"), (r"\bif not ", "if "), (r"\bif ", "if not ")]
TABLE_ENTRY = re.compile(r'\s+(exp\.\w+|"[A-Z_]+"|TokenType\.\w+)\s*:')
_NOT_DELETABLE = ("def ", "class ", "@", "#", "else", "elif", "try", "except", "finally", "return", "if ", "for ",
                  "while ", "with ")


def unified(path: str, old: str, new: str) -> str:
    body = "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        f"a/{path}", f"b/{path}", n=3))
    return f"diff --git a/{path} b/{path}\n" + body


def _mutate_line(line: str, rng: random.Random) -> str | None:
    swaps = [(p, r) for p, r in SWAPS if re.search(p, line)]
    strings = list(re.finditer(r'"([A-Za-z_ ]{2,})"', line))
    stripped = line.strip()
    choices = (["swap"] if swaps else []) + (["string"] if strings else [])
    if stripped and not stripped.startswith(_NOT_DELETABLE) and not line.rstrip().endswith((":", "(", "[", "{", ",")):
        choices.append("delete")
    if not choices:
        return None
    choice = rng.choice(choices)
    if choice == "swap":
        pattern, replacement = rng.choice(swaps)
        found = rng.choice(list(re.finditer(pattern, line)))
        return line[:found.start()] + re.sub(pattern, replacement, found.group(0)) + line[found.end():]
    if choice == "string":
        found = rng.choice(strings)
        text = found.group(1)
        return line[:found.start(1)] + text[:-1] + ("X" if text[-1] != "X" else "Y") + line[found.end(1):]
    if stripped.startswith(("return", "yield")):
        return None
    return line[: len(line) - len(line.lstrip())] + "pass\n"


def _code_mutant(root: str, coverage: dict, kind: str, rng: random.Random):
    files = [f for f in coverage["line_hits"] if f.startswith("sqlglot/") and f.endswith(".py")]
    while True:
        path = rng.choice(files)
        with open(os.path.join(root, path), encoding="utf-8", newline="") as handle:
            old = handle.read()
        lines = old.splitlines(keepends=True)
        import_time = coverage["import_time_lines"].get(path, set())
        if kind == "executed":
            spots = [n for n in coverage["line_hits"][path] if 1 <= n <= len(lines)]
        elif kind == "import":
            try:
                tree = ast.parse(old)
            except SyntaxError:
                continue
            spots = [n for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                     for n in range(f.body[0].lineno, f.end_lineno + 1) if n in import_time]
        else:  # class/module-level lines: tables, flags, constants
            spots = [n for n in import_time if 1 <= n <= len(lines)
                     and not lines[n - 1].lstrip().startswith(("def ", "class ", "@", "import ", "from "))]
            spots += [i + 1 for i, text in enumerate(lines) if TABLE_ENTRY.match(text)]
        if not spots:
            continue
        for _ in range(20):
            number = rng.choice(spots)
            line = lines[number - 1]
            new_line = _mutate_line(line, rng)
            if kind == "table" and new_line is None and TABLE_ENTRY.match(line) and line.rstrip().endswith(","):
                new_line = ""  # drop the entry
            if new_line is None or new_line == line:
                continue
            new = "".join(lines[: number - 1] + [new_line] + lines[number:])
            try:
                ast.parse(new)
            except SyntaxError:
                continue
            return path, old, new


def _fixture_mutant(root: str, coverage: dict, rng: random.Random):
    fixtures = [f for f in coverage["file_reads"] if f.startswith("tests/fixtures/") and f.endswith(".sql")]
    while True:
        path = rng.choice(fixtures)
        with open(os.path.join(root, path), encoding="utf-8", newline="") as handle:
            old = handle.read()
        lines = old.splitlines(keepends=True)
        for _ in range(50):
            number = rng.randrange(len(lines))
            text = lines[number]
            body = text.rstrip("\n")
            if len(text.strip()) < 8 or text.startswith(("#", "--")):
                continue
            i = rng.randrange(len(body))
            changed = text[:i] + text[i + 1:i + 2] + text[i:i + 1] + text[i + 2:] if i + 1 < len(body) else text[:i] + "x" + text[i:]
            if changed != text:
                return path, old, "".join(lines[:number] + [changed] + lines[number + 1:])


def general(root: str, out_dir: str, count: int, seed: int = 1, only: str | None = None) -> None:
    rng = random.Random(seed)
    coverage = read_artifact(os.path.join(ATLAS, "coverage.pkl.gz"))
    os.makedirs(out_dir, exist_ok=True)
    for made in range(count):
        if only == "import":
            (path, old, new), name = _code_mutant(root, coverage, "import", rng), f"i{made:03d}"
        else:
            draw = rng.random()
            kind = "exec" if draw < 0.55 else "table" if draw < 0.8 else "fixture"
            found = (_fixture_mutant(root, coverage, rng) if kind == "fixture"
                     else _code_mutant(root, coverage, "executed" if kind == "exec" else "table", rng))
            (path, old, new), name = found, f"m{made:03d}{kind[0]}"
        with open(os.path.join(out_dir, name + ".patch"), "w", encoding="utf-8", newline="") as handle:
            handle.write(unified(path, old, new))
    print(f"wrote {count} patches to {out_dir}")


_SPELLING = ["==", "!=", "<", "<=", ">", ">=", "is", "is not", "in", "not in"]
_SWAP_TO = {0: [1, 6], 1: [0, 7], 2: [3, 4], 3: [2, 5], 4: [5, 2], 5: [4, 3], 6: [7, 0], 7: [6, 1], 8: [9], 9: [8]}
_TOKEN = {0: r"==", 1: r"!=", 2: r"<(?!=)", 3: r"<=", 4: r">(?!=)", 5: r">=", 6: r"\bis\b(?!\s+not\b)",
          7: r"\bis\s+not\b", 8: r"(?<!not )\bin\b", 9: r"\bnot\s+in\b"}


def comparisons(root: str, tests_json: str, out_dir: str, count: int = 60, seed: int = 7) -> None:
    os.environ["RIPPLE_SQLGLOT"] = root
    from ripple import bits
    from ripple.atlas import CoverageAtlas
    from ripple.diffs import read_diff
    from ripple.impact import ImpactAnalyzer

    rng = random.Random(seed)
    atlas = read_artifact(os.path.join(ATLAS, "comparisons.pkl.gz"))
    with open(tests_json) as handle:
        tests = json.load(handle)["tests"]
    ids = [t["id"] for t in tests]
    analyzer = ImpactAnalyzer(CoverageAtlas(read_artifact(os.path.join(ATLAS, "coverage.pkl.gz")), ids), tests, root)
    sites = [k for k in atlas["results"] if k not in atlas["import_time"] and atlas["sites"][k][0] in _SWAP_TO]
    rng.shuffle(sites)
    os.makedirs(os.path.join(out_dir, "commits"), exist_ok=True)
    sources, made = {}, {}

    def offset(text, line, column):  # columns are utf-8 byte offsets
        lines = text.splitlines(keepends=True)
        return sum(len(x) for x in lines[: line - 1]) + len(lines[line - 1].encode("utf-8")[:column].decode("utf-8", errors="ignore"))

    for site in sites:
        if len(made) >= count:
            break
        path, line, column, end_line, end_column = site
        code, side = atlas["sites"][site]
        if path not in sources:
            with open(os.path.join(root, path), encoding="utf-8", newline="") as handle:
                sources[path] = handle.read()
        text = sources[path]
        node = next((x for x in ast.walk(ast.parse(text)) if isinstance(x, ast.Compare)
                     and (x.lineno, x.col_offset, x.end_lineno, x.end_col_offset) == site[1:]), None)
        if node is None or len(node.ops) != 1:
            continue
        if side and rng.random() < 0.4:
            operand = node.left if side == "L" else node.comparators[0]
            try:
                value = ast.literal_eval(operand)
            except Exception:
                continue
            if isinstance(value, bool):
                changed = not value
            elif isinstance(value, int):
                changed = value + rng.choice([-1, 1])
            elif isinstance(value, str) and value:
                changed = value.upper() if value != value.upper() else value.lower()
                changed = changed if changed != value else value + "x"
            elif isinstance(value, tuple) and value and isinstance(value[-1], int) and not isinstance(value[-1], bool):
                changed = value[:-1] + (value[-1] + 1,)
            else:
                continue
            start, end = offset(text, operand.lineno, operand.col_offset), offset(text, operand.end_lineno, operand.end_col_offset)
            new, what = text[:start] + repr(changed) + text[end:], f"literal {value!r}->{changed!r}"
        else:
            start = offset(text, node.left.end_lineno, node.left.end_col_offset)
            end = offset(text, node.comparators[0].lineno, node.comparators[0].col_offset)
            found = re.search(_TOKEN[code], text[start:end])
            if not found:
                continue
            target = rng.choice(_SWAP_TO[code])
            gap = text[start:end]
            new = text[:start] + gap[: found.start()] + _SPELLING[target] + gap[found.end():] + text[end:]
            what = f"{_SPELLING[code]} -> {_SPELLING[target]}"
        try:
            compile(new, path, "exec")
        except SyntaxError:
            continue
        patch = unified(path, text, new)
        name = f"c{len(made):03d}"
        impact = analyzer.analyze(read_diff(patch))
        chosen = range(len(ids)) if impact.everything else bits.members(impact.candidates)
        made[name] = {"site": list(site), "what": what, "candidates": [ids[i] for i in chosen]}
        with open(os.path.join(out_dir, "commits", name + ".patch"), "w", encoding="utf-8", newline="") as handle:
            handle.write(patch)
    with open(os.path.join(out_dir, "cands.json"), "w") as handle:
        json.dump(made, handle, indent=0)
    print(f"{len(made)} mutations")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "general":
        general(args[0], args[1], int(args[2]), int(args[3]) if len(args) > 3 else 1, args[4] if len(args) > 4 else None)
    else:
        comparisons(args[0], args[1], args[2], *(int(a) for a in args[3:5]))
