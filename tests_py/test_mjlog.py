"""Tests for :mod:`mjx.mjlog`.

The parser and the log self-consistency validator are pure-Python and run
without the native ``_mjx`` extension. ``MjlogReplayAgent`` is a thin
scripted-replay agent; its agent-contract tests need the engine and are
skipped when it is not built.
"""

import glob
import os

import pytest

import mjx.mjlog as _m
from mjx.mjlog import (
    DecisionType,
    MjlogGame,
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

requires_engine = pytest.mark.skipif(
    not _m._ENGINE_AVAILABLE, reason="native _mjx engine not built"
)


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
    types = sorted(tile_type(t) for t in meld.tiles)
    assert types[1] == types[0] + 1 and types[2] == types[1] + 1


def test_decode_meld_kinds_present():
    kinds = set()
    for f in MJLOGS:
        for r in parse_mjlog(f).rounds:
            for d in r.events:
                if d.meld is not None:
                    kinds.add(d.meld.kind)
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
    assert wins
    for w in wins:
        assert w.who in range(4)
        assert (w.who == w.from_who) == w.is_tsumo


def test_events_include_draws():
    game = parse_mjlog(UPLOADED)
    for r in game.rounds:
        draws = [e for e in r.events if e.type is DecisionType.DRAW]
        assert draws
        assert len(r.decisions) == len(r.events) - len(draws)


# ---------------------------------------------------------------------------
# Log self-consistency (validates the LOG, treating Tenhou as the oracle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", MJLOGS, ids=[os.path.basename(p) for p in MJLOGS])
def test_validate_corpus(path):
    report = parse_mjlog(path).validate(raise_on_error=False)
    assert report.ok, report.summary()


def test_validate_uploaded_details():
    report = MjlogGame.from_file(UPLOADED).validate()
    assert report.ok
    assert report.n_rounds == 10
    assert report.reached_game_end
    assert report.final_scores == [170, 170, 247, 413]
    assert sum(report.final_scores) * 100 == 100000


def test_partial_logs_validate_without_owari():
    partial = [p for p in MJLOGS if "no-game-end" in p or "no-round-end" in p]
    assert partial
    for path in partial:
        report = parse_mjlog(path).validate(raise_on_error=False)
        assert report.ok, report.summary()
        assert not report.reached_game_end


def test_validation_error_is_raised_on_tampered_log():
    game = parse_mjlog(UPLOADED)
    game.rounds[1].init_scores = [999, 0, 0, 0]
    with pytest.raises(MjlogValidationError):
        game.validate(raise_on_error=True)
    assert not game.validate(raise_on_error=False).ok


# ---------------------------------------------------------------------------
# MjlogReplayAgent: a scripted-replay agent and nothing more
# ---------------------------------------------------------------------------


def test_agent_script_matches_parsed_decisions():
    """The per-player replay queue is exactly the parsed positive decisions."""
    from collections import Counter

    agent = MjlogReplayAgent.from_file(UPLOADED)
    parsed = Counter(d.who for d in agent.decisions())
    for who in range(4):
        assert len(agent._queues[who]) == parsed[who]


def test_act_requires_engine_when_absent():
    agent = MjlogReplayAgent.from_file(UPLOADED)
    if not _m._ENGINE_AVAILABLE:
        with pytest.raises(RuntimeError):
            agent.act(object())


@requires_engine
def test_agent_is_a_mjx_agent():
    import mjx
    from mjx.agents import MjlogReplayAgent as FromAgents

    agent = MjlogReplayAgent.from_file(UPLOADED)
    assert isinstance(agent, mjx.Agent)
    assert FromAgents is MjlogReplayAgent  # exposed alongside the other agents


@requires_engine
def test_agent_passes_validate_agent():
    """Satisfies the same contract harness as every built-in agent."""
    from mjx.agents import validate_agent

    agent = MjlogReplayAgent.from_file(UPLOADED)
    validate_agent(agent, n_games=1, use_batch=False)
    validate_agent(agent, n_games=1, use_batch=True)


@requires_engine
def test_act_returns_a_legal_action():
    import mjx

    agent = MjlogReplayAgent.from_file(UPLOADED)
    env = mjx.MjxEnv()
    obs_dict = env.reset(1234)
    for _ in range(20):
        if env.done():
            break
        actions = {}
        for pid, obs in obs_dict.items():
            a = agent.act(obs)
            assert a in obs.legal_actions()
            actions[pid] = a
        obs_dict = env.step(actions)


# ---------------------------------------------------------------------------
# The engine's state computation (驗證狀態計算正確).
#
# To Mjx, four MjlogReplayAgents are just four agents playing: the same standard
# MjxEnv loop as any other agents (cf. tests_py/test_env.py). The wall is dealt
# from the game's seed, so the engine sees Tenhou's tiles and computes every
# state transition itself; its closing scores must match what Tenhou recorded.
# ---------------------------------------------------------------------------

_SEATS = ["0", "1", "2", "3"]


def _engine_final_scores(game):
    import mjx

    agents = {seat: MjlogReplayAgent(game) for seat in _SEATS}
    env = mjx.MjxEnv(_SEATS)
    obs = env.reset_from_tenhou_seed(game.seed)
    while not env.done():
        obs = env.step({seat: agents[seat].act(o) for seat, o in obs.items()})
    tens = env.state().to_proto().round_terminal.final_score.tens
    return [t // 100 for t in tens]  # units of 100, like the log's owari


@requires_engine
def test_reset_from_tenhou_seed_reproduces_wall():
    """The engine deals exactly the tiles Tenhou dealt for the seed."""
    import mjx

    game = parse_mjlog(UPLOADED)
    r0 = game.rounds[0]
    env = mjx.MjxEnv(_SEATS)
    env.reset_from_tenhou_seed(game.seed)
    wall = env.state().to_proto().hidden_state.wall
    # wall[52] is the dealer's first draw; wall[130] is the dora indicator.
    first_draw = next(e.tile for e in r0.events if e.type is DecisionType.DRAW)
    assert wall[52] == first_draw
    assert wall[130] == r0.dora_indicators[0]


@requires_engine
def test_engine_reproduces_uploaded_game():
    game = parse_mjlog(UPLOADED)
    assert _engine_final_scores(game) == game.final_scores


_COMPLETE = [
    p
    for p in MJLOGS
    if (g := parse_mjlog(p)).final_scores is not None and g.seed is not None
]


@requires_engine
@pytest.mark.parametrize("path", _COMPLETE, ids=[os.path.basename(p) for p in _COMPLETE])
def test_engine_reproduces_corpus(path):
    game = parse_mjlog(path)
    assert _engine_final_scores(game) == game.final_scores
