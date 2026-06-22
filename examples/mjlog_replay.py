"""Replay Tenhou ``.mjlog`` games and validate Mjx's state computation.

Two independent checks are available:

1. **Engine replay (驗引擎).**  Four :class:`mjx.mjlog.MjlogReplayAgent` instances
   replay the recorded actions against each other inside :class:`mjx.MjxEnv`.
   The tile wall is reproduced from the game's ``<SHUFFLE>`` seed
   (:meth:`mjx.MjxEnv.reset_from_tenhou_seed`), so the engine deals exactly the
   tiles Tenhou dealt and then computes every state transition (draws, melds,
   riichi, yaku/fu/score, kyotaku, honba) on its own.  Each round's engine-
   computed terminal is compared against the values recorded in the log.  A
   mismatch points at the *engine*, not the log.  Requires the native ``_mjx``
   build.

2. **Log self-consistency (驗牌譜).**  Pure-Python re-derivation of every score /
   kyotaku / honba / hand transition from the recorded deltas, checked against
   the log's own snapshots.  Needs no native build.

Usage::

    python examples/mjlog_replay.py path/to/game.mjlog [more.mjlog ...]
    python examples/mjlog_replay.py 'tests_cpp/resources/mjlog/*.mjlog'
    python examples/mjlog_replay.py --log-only 'tests_cpp/resources/mjlog/*.mjlog'
"""

import argparse
import glob
import sys

from mjx.mjlog import parse_mjlog

try:
    from mjx.mjlog import replay_in_engine

    _ENGINE = True
except Exception:  # pragma: no cover
    _ENGINE = False


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="mjlog file(s) or glob(s)")
    parser.add_argument(
        "--log-only",
        action="store_true",
        help="only check the log's self-consistency (no engine)",
    )
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.paths:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    use_engine = _ENGINE and not args.log_only
    n_ok = 0
    for path in paths:
        game = parse_mjlog(path)
        name = path.rsplit("/", 1)[-1]
        if use_engine and game.seed is not None:
            report = replay_in_engine(game, raise_on_error=False)
        else:
            report = game.validate(raise_on_error=False)
        status = "OK  " if report.ok else "FAIL"
        print(f"{status} {name}")
        print(report.summary())
        print()
        n_ok += report.ok

    mode = "engine replay" if use_engine else "log self-consistency"
    print(f"{n_ok}/{len(paths)} files validated ({mode})")
    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
