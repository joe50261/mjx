"""Tests for :mod:`mjx.mjlog` (Tenhou mjlog replay + state validation).

These tests exercise the pure-Python parsing / validation layer and therefore
run without the native ``_mjx`` extension being built.
"""

import glob
import os

import pytest

from mjx.mjlog import (
    DecisionType,
    MjlogReplayAgent,
    MjlogValidationError,
    Win,
    decode_meld,
    is_red_five,
    parse_mjlog,
    tile_type,
)

RESOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "mjlog")
UPLOADED = os.path.join(RESOURCE_DIR, "2026040201gm-00a9-0000-1c038792.mjlog")

MJLOGS = sorted(glob.glob(os.path.join(RESOURCE_DIR, "*.mjlog")))


def test_resources_present():
    assert MJLOGS, "no mjlog resources found"
    assert os.path.exists(UPLOADED)


# ---------------------------------------------------------------------------
# Tile / meld helpers
# ---------------------------------------------------------------------------


def test_tile_type():
    assert tile_type(0) == 0  # 1m
    assert tile_type(135) == 33  # red dragon
    assert tile_type(16) == 4 and tile_type(17) == 4  # 5m


def test_is_red_five():
    assert is_red_five(16) and is_red_five(52) and is_red_five(88)
    assert not is_red_five(17)


def test_decode_meld_pon():
    meld = decode_meld(2, 51305)
    assert meld.kind == "pon"
    assert tuple(sorted(meld.tiles)) == (132, 133, 134)  # red dragons
    assert meld.from_who == 3  # stolen from player 3's discard
    assert meld.stolen_tile == meld.tiles[meld.called]


def test_decode_meld_chi():
    meld = decode_meld(2, 26719)
    assert meld.kind == "chi"
    # a chi is three consecutive tile types of the same suit
    types = sorted(tile_type(t) for t in meld.tiles)
    assert types[1] == types[0] + 1 and types[2] == types[1] + 1


def test_decode_meld_kinds_present():
    kinds = set()
    for f in MJLOGS:
        for r in parse_mjlog(f).rounds:
            for d in r.events:
                if d.meld is not None:
                    kinds.add(d.meld.kind)
    # the corpus contains every call type
    assert {"chi", "pon", "open_kan", "closed_kan", "added_kan"} <= kinds


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_uploaded_metadata():
    game = parse_mjlog(UPLOADED)
    assert game.version == "2.3"
    assert game.players == ["かなきー", "東雲ハヤテ", "麺屋真島", "おがこーさん"]
    assert game.dans == [16, 17, 18, 17]
    assert len(game.rounds) == 10
    assert game.final_scores == [170, 170, 247, 413]
    assert sum(game.final_scores) == 1000  # 100000 points total


def test_parse_from_str_matches_file():
    with open(UPLOADED, encoding="utf-8") as fh:
        xml = fh.read()
    assert len(parse_mjlog(xml).rounds) == len(parse_mjlog(UPLOADED).rounds)


def test_round_opening_hands_are_13():
    for r in parse_mjlog(UPLOADED).rounds:
        for hand in r.init_hands:
            assert len(hand) == 13


def test_results_are_typed():
    game = parse_mjlog(UPLOADED)
    wins = [res for r in game.rounds for res in r.results if isinstance(res, Win)]
    assert wins  # the uploaded game has wins
    for w in wins:
        assert w.who in range(4)
        assert (w.who == w.from_who) == w.is_tsumo


# ---------------------------------------------------------------------------
# State validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", MJLOGS, ids=[os.path.basename(p) for p in MJLOGS])
def test_validate_corpus(path):
    """Every recorded score / kyotaku / hand transition reconciles."""
    report = parse_mjlog(path).validate(raise_on_error=False)
    assert report.ok, report.summary()


def test_validate_uploaded_details():
    report = MjlogReplayAgent.from_file(UPLOADED).validate()
    assert report.ok
    assert report.n_rounds == 10
    assert report.reached_game_end
    assert report.final_scores == [170, 170, 247, 413]
    # closing scores (incl. the leftover riichi stick) conserve 100000 points
    assert sum(report.final_scores) * 100 == 100000


def test_point_conservation_whole_corpus():
    for path in MJLOGS:
        report = parse_mjlog(path).validate(raise_on_error=False)
        assert report.ok, f"{os.path.basename(path)}: {report.summary()}"


def test_partial_logs_validate_without_owari():
    # The "-no-game-end" / "-no-round-end" logs are intentionally truncated and
    # never reach <owari>; they must still reconcile (sticks stay on the table).
    partial = [p for p in MJLOGS if "no-game-end" in p or "no-round-end" in p]
    assert partial
    for path in partial:
        report = parse_mjlog(path).validate(raise_on_error=False)
        assert report.ok, report.summary()
        assert not report.reached_game_end


def test_validation_error_is_raised_on_tampered_log():
    game = parse_mjlog(UPLOADED)
    # Corrupt an opening score so the continuity check must fail.
    game.rounds[1].init_scores = [999, 0, 0, 0]
    with pytest.raises(MjlogValidationError):
        game.validate(raise_on_error=True)
    report = game.validate(raise_on_error=False)
    assert not report.ok


# ---------------------------------------------------------------------------
# Replay decisions
# ---------------------------------------------------------------------------


def test_decisions_stream():
    agent = MjlogReplayAgent.from_file(UPLOADED)
    decisions = agent.decisions()
    assert decisions
    # draws are environment events, not player decisions
    assert all(d.type is not DecisionType.DRAW for d in decisions)
    # every discard-like decision carries a tile id
    for d in decisions:
        if d.type in (DecisionType.DISCARD, DecisionType.TSUMOGIRI, DecisionType.RIICHI):
            assert d.tile is not None and 0 <= d.tile < 136


def test_events_include_draws():
    game = parse_mjlog(UPLOADED)
    for r in game.rounds:
        draws = [e for e in r.events if e.type is DecisionType.DRAW]
        assert draws  # every round has draws
        # the player choices are exactly events minus draws
        assert len(r.decisions) == len(r.events) - len(draws)


def test_act_requires_engine_when_absent():
    import mjx.mjlog as m

    agent = MjlogReplayAgent.from_file(UPLOADED)
    if not m._ENGINE_AVAILABLE:
        with pytest.raises(RuntimeError):
            agent.act(object())


# ---------------------------------------------------------------------------
# Engine reconstruction / replay
# ---------------------------------------------------------------------------

import mjx.mjlog as _m  # noqa: E402

requires_engine = pytest.mark.skipif(
    not _m._ENGINE_AVAILABLE, reason="native _mjx engine not built"
)


def test_reconstructed_wall_is_permutation():
    # Engine-independent: the wall we hand the engine must be a 0..135 permutation.
    game = parse_mjlog(UPLOADED)
    for r in game.rounds:
        wall = _m._reconstruct_wall(r)
        assert sorted(wall) == list(range(136))


def test_state_json_round_trips_to_dict():
    game = parse_mjlog(UPLOADED)
    import json

    d = json.loads(_m.round_to_state_json(game.rounds[0]))
    assert len(d["hiddenState"]["wall"]) == 136
    assert d["publicObservation"]["initScore"]["tens"] == [25000, 25000, 25000, 25000]
    assert d["publicObservation"]["events"][0]["type"] == "EVENT_TYPE_DRAW"


@requires_engine
def test_engine_replays_uploaded_game():
    """The native engine recomputes the whole Tenhou game from wall + actions."""
    report = MjlogReplayAgent.from_file(UPLOADED).validate_with_engine()
    # Every round replays and the engine reproduces Tenhou's action stream.
    assert report.n_replayed == report.n_rounds == 10, report.summary()
    assert report.n_action_match == report.n_rounds, report.summary()


@requires_engine
@pytest.mark.parametrize("path", MJLOGS, ids=[os.path.basename(p) for p in MJLOGS])
def test_engine_replays_corpus(path):
    """Every round of every bundled game replays through the engine and the
    engine's regenerated action stream matches Tenhou exactly."""
    report = MjlogReplayAgent.from_file(path).validate_with_engine(raise_on_error=False)
    assert report.n_replayed == report.n_rounds, report.summary()
    assert report.n_action_match == report.n_rounds, report.summary()
