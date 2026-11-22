"""Merge the raw coverage records into atlas/coverage.pkl.gz, and add the static facts.

    FORGE_ISOLATED=<dir> FORGE_IMPORT_TRACE=<json> FORGE_MODULE_LOADS=<json> \\
        python -m forge.assemble_coverage <raw dir> <tests.json> <atlas/coverage.pkl.gz> [sqlglot checkout]

Records: tests-*.jsonl (pytest_coverage), sub-*.jsonl (sitecustomize), iso-*.jsonl
(isolated_coverage). Static facts, from the pinned checkout: the import graph (module-level and
in-function imports), the read sites of every name assigned at class or module level, and
dynamic imports with computed module names.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import sys

from ripple.atlas import write_artifact
from ripple.syntax import runtime_imports

_BLOCK_STATEMENTS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.With, ast.Try)


class _Merger:
    def __init__(self, test_ids: list[str]):
        self.bit = {test_id: 1 << i for i, test_id in enumerate(test_ids)}
        self.line_hits: dict[str, dict[int, int]] = {}
        self.file_reads: dict[str, int] = {}
        self.dir_scans: dict[str, int] = {}
        self.instantiated: dict[str, int] = {}
        self.import_time_lines: dict[str, set[int]] = {}
        self.class_setup: dict[str, dict[str, set[int]]] = {}
        self.class_members: dict[str, int] = {}
        self.recorded: set[str] = set()
        self.unknown: list[str] = []

    def hit(self, tests: int, lines_by_path: dict) -> None:
        for path, lines in lines_by_path.items():
            hits = self.line_hits.setdefault(path, {})
            for line in lines:
                hits[line] = hits.get(line, 0) | tests

    def add(self, row: dict) -> None:
        test_id = row["id"]
        if test_id in ("<import>", "__import__"):
            for path, lines in row["lines"].items():
                self.import_time_lines.setdefault(path, set()).update(lines)
            return
        if test_id not in self.bit:
            self.unknown.append(test_id)
            return
        tests = self.bit[test_id]
        self.recorded.add(test_id)
        self.hit(tests, row["lines"])
        if "setup" in row:
            klass = test_id.rsplit("::", 1)[0]
            self.class_members[klass] = self.class_members.get(klass, 0) | tests
            setup = self.class_setup.setdefault(klass, {})
            for path, lines in row["setup"].items():
                setup.setdefault(path, set()).update(lines)
        for field, table in (("opens", self.file_reads), ("dirs", self.dir_scans), ("classes", self.instantiated)):
            for key in row.get(field, ()):
                table[key] = table.get(key, 0) | tests

    def finish(self) -> None:
        # a class setup (setUpClass runs in its first test's setup) belongs to all its tests
        for klass, setup in self.class_setup.items():
            self.hit(self.class_members[klass], setup)


def _python_files(root: str) -> list[str]:
    found = []
    for top in ("sqlglot", "tests"):
        for directory, _, names in os.walk(os.path.join(root, top)):
            found += [os.path.relpath(os.path.join(directory, n), root).replace(os.sep, "/") for n in names if n.endswith(".py")]
    return found


def _module(path: str) -> tuple[str, bool]:
    name = path[:-3].replace("/", ".")
    return (name[: -len(".__init__")], True) if name.endswith(".__init__") else (name, False)


def import_graph(root: str, files: list[str]):
    """(imported_by: path -> module-level importers, lazy_imports: path -> in-function sites)."""
    path_of = {_module(f)[0]: f for f in files}
    imported_by: dict[str, list[str]] = {}
    lazy: dict[str, set] = {}
    for path in files:
        module, is_package = _module(path)
        with open(os.path.join(root, path), "rb") as handle:
            source = handle.read()
        targets = set()
        for name, in_function, first, last in runtime_imports(source, module, is_package):
            parts = name.split(".")
            target = next((path_of[".".join(parts[:i])] for i in range(len(parts), 0, -1) if ".".join(parts[:i]) in path_of), None)
            if target is None or target == path:
                continue
            if in_function:
                lazy.setdefault(target, set()).add((path, first, last))
            else:
                targets.add(target)
        for target in targets:
            imported_by.setdefault(target, []).append(path)
    return {k: sorted(v) for k, v in imported_by.items()}, {k: sorted(v) for k, v in lazy.items()}


def _span(statement: ast.stmt) -> tuple[int, int]:
    if isinstance(statement, _BLOCK_STATEMENTS):
        body = getattr(statement, "body", None)
        return statement.lineno, max(statement.lineno, (body[0].lineno - 1) if body else statement.lineno)
    return statement.lineno, statement.end_lineno


def name_reads(trees: dict) -> dict:
    """name -> [(path, first, last)] of the statements reading a name assigned at class or
    module level (as a variable, an attribute, or an identifier-like string)."""
    assigned = set()
    for tree in trees.values():
        for namespace in [tree] + [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for statement in namespace.body:
                targets = (statement.targets if isinstance(statement, ast.Assign)
                           else [statement.target] if isinstance(statement, ast.AnnAssign) else [])
                assigned.update(t.id for t in targets if isinstance(t, ast.Name))
    reads: dict[str, set] = {}
    for path, tree in trees.items():
        def visit(node, statement):
            for child in ast.iter_child_nodes(node):
                current = child if isinstance(child, ast.stmt) else statement
                name = None
                if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                    name = child.attr
                elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    name = child.id
                elif isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.isidentifier():
                    name = child.value
                if name in assigned and current is not None:
                    reads.setdefault(name, set()).add((path,) + _span(current))
                visit(child, current)

        visit(tree, None)
    return {k: sorted(v) for k, v in reads.items()}


def dynamic_imports(trees: dict) -> dict:
    """Static prefix of a computed module name (``import_module(f"sqlglot.dialects.{x}")``) ->
    [(path, first, last)] of the call sites; "" for a fully computed name."""
    sites: dict[str, set] = {}
    for path, tree in trees.items():
        for statement in ast.walk(tree):
            if not isinstance(statement, ast.stmt) or isinstance(statement, _BLOCK_STATEMENTS):
                continue
            for call in ast.walk(statement):
                callee = getattr(call, "func", None)
                if not (isinstance(call, ast.Call) and getattr(callee, "attr", getattr(callee, "id", None))
                        in ("import_module", "__import__") and call.args):
                    continue
                argument = call.args[0]
                prefix = None
                if isinstance(argument, ast.JoinedStr) and argument.values and isinstance(argument.values[0], ast.Constant):
                    prefix = argument.values[0].value
                elif (isinstance(argument, ast.BinOp) and isinstance(argument.left, ast.Constant)
                      and isinstance(argument.left.value, str)):
                    prefix = argument.left.value
                elif not isinstance(argument, ast.Constant):
                    prefix = ""
                if prefix is not None:
                    sites.setdefault(prefix, set()).add((path, statement.lineno, statement.end_lineno))
    return {k: sorted(v) for k, v in sites.items()}


def main(raw_dir: str, tests_json: str, output: str, root: str = "/opt/sqlglot") -> None:
    with open(tests_json) as handle:
        test_ids = [t["id"] for t in json.load(handle)["tests"]]
    merger = _Merger(test_ids)
    isolated = os.environ.get("FORGE_ISOLATED")
    record_files = (sorted(glob.glob(f"{raw_dir}/tests-*.jsonl")) + sorted(glob.glob(f"{raw_dir}/sub-*.jsonl"))
                    + (sorted(glob.glob(f"{isolated}/iso-*.jsonl")) if isolated else []))
    for record_file in record_files:
        with open(record_file) as handle:
            for row in handle:
                merger.add(json.loads(row))
    merger.finish()
    missing = [t for t in test_ids if t not in merger.recorded]
    print(f"recorded {len(merger.recorded)}/{len(test_ids)} tests; unknown ids {len(merger.unknown)}; "
          f"not recorded {missing[:10]}", file=sys.stderr)

    module_loads: dict[str, int] = {}
    if os.environ.get("FORGE_MODULE_LOADS"):
        with open(os.environ["FORGE_MODULE_LOADS"]) as handle:
            for test_id, modules in json.load(handle).items():
                if test_id in merger.bit:
                    for module in modules:
                        module_loads[module] = module_loads.get(module, 0) | merger.bit[test_id]
    import_time_calls = {}
    if os.environ.get("FORGE_IMPORT_TRACE"):
        with open(os.environ["FORGE_IMPORT_TRACE"]) as handle:
            import_time_calls = {path: [(first, last, tuple(tuple(b) for b in bodies)) for first, last, bodies in spans]
                                 for path, spans in json.load(handle).items()}

    files = _python_files(root)
    imported_by, lazy_imports = import_graph(root, files)
    trees = {}
    for path in files:
        try:
            with open(os.path.join(root, path), "rb") as handle:
                trees[path] = ast.parse(handle.read())
        except SyntaxError:
            continue
    write_artifact(output, {
        "test_ids": test_ids,
        "line_hits": merger.line_hits,
        "file_reads": merger.file_reads,
        "dir_scans": merger.dir_scans,
        "instantiated": merger.instantiated,
        "module_loads": module_loads,
        "import_time_lines": merger.import_time_lines,
        "import_time_calls": import_time_calls,
        "imported_by": imported_by,
        "lazy_imports": lazy_imports,
        "name_reads": name_reads(trees),
        "dynamic_imports": dynamic_imports(trees),
    }, level=9)
    print(f"coverage atlas: {len(merger.line_hits)} files, {sum(map(len, merger.line_hits.values()))} lines", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:5])
