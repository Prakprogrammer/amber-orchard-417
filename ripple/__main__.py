"""``python -m ripple``: one JSON line per commit, {"commit": id, "workers": [[test, ...], ...]}."""

import sys
import traceback

from .pipeline import Pipeline, list_everything
from .settings import Settings


def main() -> None:
    settings = Settings.from_environment()
    printed = False
    try:
        lines = Pipeline(settings).run()
        sys.stdout.write("".join(line + "\n" for line in lines))
        printed = True
    except BaseException:  # noqa: whatever broke, still answer (listing everything never misses)
        traceback.print_exc()
        if not printed:
            sys.stdout.write("".join(line + "\n" for line in list_everything(settings)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
