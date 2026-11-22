"""Caches made once, at image build time, each in its own fresh process:

    python -m ripple.prepare tpc-rows <output>           (ripple.tpc_rows)
    python -m ripple.prepare dialect-families <output>   (ripple.reductions.build_family_table)
"""

import os
import pickle
import sys


def main(argv: list[str]) -> None:
    what, output = argv[0], os.path.abspath(argv[1])  # (the builders change directory)
    root = os.environ.get("RIPPLE_SQLGLOT", "/opt/sqlglot")
    if what == "tpc-rows":
        from .tpc_rows import build

        build(output, root)
    elif what == "dialect-families":
        from .reductions import build_family_table

        table = build_family_table(root)
        with open(output, "wb") as handle:
            pickle.dump(table, handle)
        print(f"dialect families: {sum(v is not None for v in table.values())} of {len(table)} modules", file=sys.stderr)
    else:
        raise SystemExit(f"unknown cache {what!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
