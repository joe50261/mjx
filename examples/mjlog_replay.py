"""Replay Tenhou ``.mjlog`` games and validate the state computation.

Usage::

    python examples/mjlog_replay.py path/to/game.mjlog [more.mjlog ...]
    python examples/mjlog_replay.py 'tests_py/resources/mjlog/*.mjlog'

For each file it parses the log, re-derives every score / kyotaku / honba /
hand transition and checks it against the values recorded by Tenhou.  This uses
only :mod:`mjx.mjlog`, which is pure Python, so it works even when the native
Mjx engine is not built.

When the native engine *is* available, the same :class:`mjx.mjlog.MjlogReplay
Agent` is an ``mjx.Agent`` and can be fed to ``mjx.MjxEnv`` to re-run the game
through the engine and compare the recomputed observations against the log.
"""

import glob
import sys

from mjx.mjlog import MjlogReplayAgent


def main(argv):
    if not argv:
        print(__doc__)
        return 1

    paths = []
    for pattern in argv:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    n_ok = 0
    for path in paths:
        agent = MjlogReplayAgent.from_file(path)
        report = agent.validate(raise_on_error=False)
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
