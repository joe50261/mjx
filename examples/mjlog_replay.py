"""Parse Tenhou ``.mjlog`` games and validate the log's self-consistency.

Usage::

    python examples/mjlog_replay.py path/to/game.mjlog [more.mjlog ...]
    python examples/mjlog_replay.py 'tests_py/resources/mjlog/*.mjlog'

For each file it parses the log and re-derives every score / kyotaku / honba /
hand transition, checking it against the values Tenhou recorded. This is pure
Python (``mjx.mjlog``) and needs no native build.

``mjx.mjlog.MjlogReplayAgent`` is a separate, thin :class:`mjx.Agent` that
replays the recorded actions (``act(obs)`` returns the logged action); it is
not exercised here.
"""

import glob
import sys

from mjx.mjlog import parse_mjlog


def main(argv):
    if not argv:
        print(__doc__)
        return 1

    paths = []
    for pattern in argv:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    n_ok = 0
    for path in paths:
        report = parse_mjlog(path).validate(raise_on_error=False)
        name = path.rsplit("/", 1)[-1]
        status = "OK  " if report.ok else "FAIL"
        print(f"{status} {name}")
        print(report.summary())
        print()
        n_ok += report.ok

    print(f"{n_ok}/{len(paths)} files validated")
    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
