"""Replay and validation for Tenhou ``.mjlog`` game logs.

This module provides a self-contained (standard-library only) reader for
Tenhou's ``mjlog`` XML format together with a replay state machine and a
validator that re-derives every score transition of a game and checks it
against the values recorded in the log.  It is used by
:class:`MjlogReplayAgent` (see below) to replay a recorded game through
:class:`mjx.MjxEnv`, but the parsing / validation layer does **not** depend on
the native ``_mjx`` extension, so it can be used to verify that the state
(score / kyotaku / honba / hand) bookkeeping is computed correctly even when
Mjx is not built.

The Tenhou ``mjlog`` format
---------------------------
A game is a flat list of XML elements::

    <SHUFFLE seed=".."/>          # RNG seed (unused for replay; draws are explicit)
    <GO type=".." lobby=".."/>    # game-type flags
    <UN n0=".." .. dan=".." rate=".." sx=".."/>   # players (%-encoded names)
    <TAIKYOKU oya="0"/>           # initial dealer
    <INIT seed="rnd,honba,riichi,dice1,dice2,dora_ind" ten=".." oya=".."
          hai0=".." hai1=".." hai2=".." hai3=".."/>   # start of a round
    <T123/> <U123/> <V123/> <W123/>   # draw by player 0/1/2/3 (tile id 0..135)
    <D123/> <E123/> <F123/> <G123/>   # discard by player 0/1/2/3
    <N who=".." m=".."/>          # call (chi/pon/kan) -- m is a bit field
    <REACH who=".." step="1"/>    # riichi declaration (before the discard)
    <REACH who=".." ten=".." step="2"/>   # riichi confirmed (-1000, +1 stick)
    <DORA hai=".."/>              # a new dora indicator is revealed
    <AGARI .. sc="s0,d0,s1,d1,s2,d2,s3,d3" ../>   # a win
    <RYUUKYOKU type=".." sc=".." ../>             # an exhaustive / abortive draw

Tile ids are ``0..135``; ``tile_type = tile_id // 4`` (``0..33``).  Scores in
the log are in units of 100 points (``250`` == 25000).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

__all__ = [
    "MjlogReplayAgent",
    "MjlogGame",
    "RoundLog",
    "Meld",
    "Win",
    "ExhaustiveDraw",
    "Decision",
    "DecisionType",
    "ValidationReport",
    "EngineReplayReport",
    "EngineRoundResult",
    "MjlogParseError",
    "MjlogValidationError",
    "parse_mjlog",
    "replay_in_engine",
    "decode_meld",
    "tile_type",
    "is_red_five",
]

# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

#: Tenhou tile ids for the three red fives (5m, 5p, 5s).
RED_FIVES = (16, 52, 88)

_TILE_TYPE_NAMES = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + ["ew", "sw", "ww", "nw", "wd", "gd", "rd"]
)


def tile_type(tile_id: int) -> int:
    """Return the tile *type* (0..33) for a Tenhou tile *id* (0..135)."""
    return tile_id // 4


def is_red_five(tile_id: int) -> bool:
    """Return ``True`` if ``tile_id`` is one of the three red-five tiles."""
    return tile_id in RED_FIVES


def tile_name(tile_id: int) -> str:
    """Human readable name, e.g. ``5m`` or ``0m`` for the red five of man."""
    name = _TILE_TYPE_NAMES[tile_type(tile_id)]
    if is_red_five(tile_id):
        return "0" + name[1:]
    return name


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MjlogParseError(ValueError):
    """Raised when an ``mjlog`` document cannot be parsed."""


class MjlogValidationError(AssertionError):
    """Raised when a replayed game does not match the recorded state."""


# ---------------------------------------------------------------------------
# Meld decoding (Tenhou ``m`` bit field)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Meld:
    """A decoded call (naki).

    ``kind`` is one of ``"chi"``, ``"pon"``, ``"open_kan"``, ``"closed_kan"``
    or ``"added_kan"``.  ``tiles`` are the tile ids forming the meld (4 for a
    kan), ``called`` is the index into ``tiles`` of the stolen/added tile and
    ``from_who`` is the *absolute* seat the tile was taken from (equal to
    ``who`` for a closed kan).
    """

    who: int
    kind: str
    tiles: Tuple[int, ...]
    called: int
    from_who: int
    m: int = 0  # the raw Tenhou meld bit field (mjx uses the same encoding)

    @property
    def stolen_tile(self) -> int:
        return self.tiles[self.called]


def decode_meld(who: int, data: Union[int, str]) -> Meld:
    """Decode a Tenhou meld bit field (the ``m`` attribute of ``<N>``).

    The algorithm follows the well known reference decoder
    (https://github.com/NegativeMjark/tenhou-log).  ``from_who`` is resolved
    from the relative offset stored in the low two bits into an absolute seat.
    """
    data = int(data)
    rel_from = data & 0x3  # 0=self, 1=shimocha(right), 2=toimen, 3=kamicha(left)
    from_who = (who + rel_from) % 4

    if data & 0x4:  # chi
        t0, t1, t2 = (data >> 3) & 0x3, (data >> 5) & 0x3, (data >> 7) & 0x3
        base_and_called = data >> 10
        called = base_and_called % 3
        base = base_and_called // 3
        base = (base // 7) * 9 + base % 7
        tiles = (t0 + 4 * (base + 0), t1 + 4 * (base + 1), t2 + 4 * (base + 2))
        return Meld(who, "chi", tiles, called, from_who, data)

    if data & 0x18:  # pon or added kan
        unused = (data >> 5) & 0x3
        t0, t1, t2 = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))[unused]
        base_and_called = data >> 9
        called = base_and_called % 3
        base = base_and_called // 3
        if data & 0x8:  # pon
            tiles = (t0 + 4 * base, t1 + 4 * base, t2 + 4 * base)
            return Meld(who, "pon", tiles, called, from_who, data)
        # added kan (chakan): the 4th tile is added to an existing pon
        tiles = (t0 + 4 * base, t1 + 4 * base, t2 + 4 * base, unused + 4 * base)
        return Meld(who, "added_kan", tiles, 3, from_who, data)

    # kan (closed when taken from self, open otherwise)
    base_and_called = data >> 8
    called = base_and_called % 4
    base = base_and_called // 4
    tiles = (4 * base, 1 + 4 * base, 2 + 4 * base, 3 + 4 * base)
    kind = "closed_kan" if rel_from == 0 else "open_kan"
    return Meld(who, kind, tiles, called, from_who, data)


# ---------------------------------------------------------------------------
# Decisions (player choices) and round results
# ---------------------------------------------------------------------------


class DecisionType(Enum):
    """The kind of event in a round.

    ``DRAW`` is an environment event (a tile coming off the wall); the rest are
    player choices and mirror the Mjx ``ActionType`` names.
    """

    DRAW = "draw"
    DISCARD = "discard"
    TSUMOGIRI = "tsumogiri"
    RIICHI = "riichi"
    CHI = "chi"
    PON = "pon"
    OPEN_KAN = "open_kan"
    CLOSED_KAN = "closed_kan"
    ADDED_KAN = "added_kan"
    TSUMO = "tsumo"
    RON = "ron"
    ABORTIVE_DRAW_NINE_TERMINALS = "kyuushu"


@dataclass(frozen=True)
class Decision:
    """A single recorded decision, used to drive replay."""

    who: int
    type: DecisionType
    tile: Optional[int] = None
    meld: Optional[Meld] = None


@dataclass
class Win:
    """An ``<AGARI>`` element."""

    who: int
    from_who: int
    hand: List[int]
    winning_tile: int
    fu: int
    points: int
    limit: int  # 0=normal, 1=mangan, 2=haneman, 3=baiman, 4=yakuman ...
    yaku: List[Tuple[int, int]]  # (yaku id, han) pairs
    dora_indicators: List[int]
    ura_dora_indicators: List[int]
    honba: int
    riichi_sticks: int
    score_before: List[int]
    score_delta: List[int]

    @property
    def is_tsumo(self) -> bool:
        return self.who == self.from_who


@dataclass
class ExhaustiveDraw:
    """A ``<RYUUKYOKU>`` element (exhaustive or abortive draw)."""

    type: Optional[str]  # None / "yao9" / "nm" / "kan4" / "reach4" / "kaze4"
    tenpai: List[bool]
    honba: int
    riichi_sticks: int
    score_before: List[int]
    score_delta: List[int]


# ---------------------------------------------------------------------------
# Round / game containers
# ---------------------------------------------------------------------------


@dataclass
class RoundLog:
    """One round (kyoku) of a game."""

    round: int  # 0 == East-1, 4 == South-1, ...
    honba: int
    riichi_sticks: int
    dealer: int
    dice: Tuple[int, int]
    dora_indicators: List[int]  # grows as kans reveal new indicators
    init_scores: List[int]  # in units of 100
    init_hands: List[List[int]]
    # The ordered event stream of the round, including draws (``DecisionType.
    # DRAW``).  ``decisions`` (below) is the player-choice view of this list.
    events: List[Decision] = field(default_factory=list)
    results: List[Union[Win, ExhaustiveDraw]] = field(default_factory=list)
    # (who, ten) for every *confirmed* riichi (``<REACH step="2">``). A riichi
    # that is ron'd before confirmation has no step-2 element and therefore
    # places no 1000-point stick, so it is intentionally absent here.
    riichi_confirmations: List[Tuple[int, List[int]]] = field(default_factory=list)

    @property
    def decisions(self) -> List[Decision]:
        """The player choices of the round (the event stream minus draws)."""
        return [e for e in self.events if e.type is not DecisionType.DRAW]

    @property
    def is_dealer_win(self) -> bool:
        return any(isinstance(r, Win) and r.who == self.dealer for r in self.results)


@dataclass
class MjlogGame:
    """A parsed Tenhou game log."""

    version: str
    players: List[str]
    dans: List[int]
    rates: List[float]
    sexes: List[str]
    game_type: int
    rounds: List[RoundLog] = field(default_factory=list)
    final_scores: Optional[List[int]] = None  # units of 100
    final_points: Optional[List[float]] = None  # placement points (uma/oka)
    # The ``<SHUFFLE seed="...">`` value ("mt19937ar-sha512-n288-base64,<b64>").
    # Reproduces the exact tile wall Tenhou dealt; used to drive the engine.
    seed: Optional[str] = None

    # -- convenience constructors -------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "MjlogGame":
        return parse_mjlog(path)

    @classmethod
    def from_str(cls, xml: str) -> "MjlogGame":
        return parse_mjlog(xml)

    # -- validation ---------------------------------------------------------
    def validate(self, raise_on_error: bool = True) -> "ValidationReport":
        """Replay the game and validate every recorded state transition."""
        return validate_game(self, raise_on_error=raise_on_error)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_DRAW_LETTERS = "TUVW"
_DISCARD_LETTERS = "DEFG"
_MOVE_RE = re.compile(r"^([TUVWDEFG])(\d+)$")


def _ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",")] if text else []


def parse_mjlog(source: str) -> MjlogGame:
    """Parse a Tenhou ``mjlog`` from a file path or an XML string."""
    looks_like_xml = source.lstrip()[:1] == "<"
    try:
        if looks_like_xml:
            root = ET.fromstring(source)
        else:
            root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:  # pragma: no cover - defensive
        raise MjlogParseError(str(exc)) from exc

    if root.tag != "mjloggm":
        raise MjlogParseError(f"unexpected root element: <{root.tag}>")

    game = MjlogGame(
        version=root.attrib.get("ver", ""),
        players=["", "", "", ""],
        dans=[0, 0, 0, 0],
        rates=[0.0, 0.0, 0.0, 0.0],
        sexes=["", "", "", ""],
        game_type=0,
    )

    cur: Optional[RoundLog] = None
    last_draw: List[Optional[int]] = [None, None, None, None]
    pending_riichi: List[bool] = [False, False, False, False]
    last_drawer: Optional[int] = None

    def flush(round_log: Optional[RoundLog]) -> None:
        if round_log is not None:
            game.rounds.append(round_log)

    for el in root:
        tag = el.tag
        attr = el.attrib

        move = _MOVE_RE.match(tag)
        if move is not None and cur is not None:
            letter, num = move.group(1), int(move.group(2))
            if letter in _DRAW_LETTERS:
                who = _DRAW_LETTERS.index(letter)
                last_draw[who] = num
                last_drawer = who
                cur.events.append(Decision(who, DecisionType.DRAW, tile=num))
            else:  # discard
                who = _DISCARD_LETTERS.index(letter)
                if pending_riichi[who]:
                    dtype = DecisionType.RIICHI
                    pending_riichi[who] = False
                elif last_draw[who] == num:
                    dtype = DecisionType.TSUMOGIRI
                else:
                    dtype = DecisionType.DISCARD
                cur.events.append(Decision(who, dtype, tile=num))
                last_draw[who] = None
            continue

        if tag == "SHUFFLE":
            game.seed = attr.get("seed")
        elif tag == "GO":
            game.game_type = int(attr.get("type", 0))
        elif tag == "UN" and "n0" in attr:
            for i in range(4):
                if f"n{i}" in attr:
                    game.players[i] = unquote(attr[f"n{i}"])
            if "dan" in attr:
                game.dans = _ints(attr["dan"])
            if "rate" in attr:
                game.rates = [float(x) for x in attr["rate"].split(",")]
            if "sx" in attr:
                game.sexes = attr["sx"].split(",")
        elif tag == "TAIKYOKU":
            pass  # initial dealer; INIT carries per-round dealer
        elif tag == "INIT":
            flush(cur)
            seed = _ints(attr["seed"])
            cur = RoundLog(
                round=seed[0],
                honba=seed[1],
                riichi_sticks=seed[2],
                dealer=int(attr["oya"]),
                dice=(seed[3], seed[4]),
                dora_indicators=[seed[5]],
                init_scores=_ints(attr["ten"]),
                init_hands=[_ints(attr[f"hai{i}"]) for i in range(4)],
            )
            last_draw = [None, None, None, None]
            pending_riichi = [False, False, False, False]
            last_drawer = None
        elif tag == "N" and cur is not None:
            meld = decode_meld(int(attr["who"]), attr["m"])
            cur.events.append(Decision(meld.who, DecisionType[meld.kind.upper()], meld=meld))
        elif tag == "REACH" and cur is not None:
            who = int(attr["who"])
            if attr.get("step") == "1":
                pending_riichi[who] = True
            elif attr.get("step") == "2":
                # The riichi is confirmed: 1000 points are paid now.
                cur.riichi_confirmations.append((who, _ints(attr.get("ten", ""))))
        elif tag == "DORA" and cur is not None:
            cur.dora_indicators.append(int(attr["hai"]))
        elif tag == "AGARI" and cur is not None:
            sc = _ints(attr["sc"])
            ten = _ints(attr["ten"])
            ba = _ints(attr.get("ba", "0,0"))
            cur.events.append(
                Decision(
                    int(attr["who"]),
                    (
                        DecisionType.TSUMO
                        if int(attr["who"]) == int(attr["fromWho"])
                        else DecisionType.RON
                    ),
                    tile=int(attr["machi"]),
                )
            )
            cur.results.append(
                Win(
                    who=int(attr["who"]),
                    from_who=int(attr["fromWho"]),
                    hand=_ints(attr["hai"]),
                    winning_tile=int(attr["machi"]),
                    fu=ten[0] if ten else 0,
                    points=ten[1] if len(ten) > 1 else 0,
                    limit=ten[2] if len(ten) > 2 else 0,
                    yaku=_pairs(_ints(attr.get("yaku", ""))),
                    dora_indicators=_ints(attr.get("doraHai", "")),
                    ura_dora_indicators=_ints(attr.get("doraHaiUra", "")),
                    honba=ba[0],
                    riichi_sticks=ba[1],
                    score_before=sc[0::2],
                    score_delta=sc[1::2],
                )
            )
            _maybe_owari(game, attr)
        elif tag == "RYUUKYOKU" and cur is not None:
            sc = _ints(attr["sc"])
            ba = _ints(attr.get("ba", "0,0"))
            tenpai = [f"hai{i}" in attr for i in range(4)]
            rtype = attr.get("type")
            if rtype == "yao9" and last_drawer is not None:
                cur.events.append(Decision(last_drawer, DecisionType.ABORTIVE_DRAW_NINE_TERMINALS))
            cur.results.append(
                ExhaustiveDraw(
                    type=rtype,
                    tenpai=tenpai,
                    honba=ba[0],
                    riichi_sticks=ba[1],
                    score_before=sc[0::2],
                    score_delta=sc[1::2],
                )
            )
            _maybe_owari(game, attr)
        # SHUFFLE / BYE / unknown tags are ignored.

    flush(cur)
    return game


def _pairs(values: List[int]) -> List[Tuple[int, int]]:
    return [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]


def _maybe_owari(game: MjlogGame, attr: Dict[str, str]) -> None:
    if "owari" not in attr:
        return
    ow = attr["owari"].split(",")
    game.final_scores = [int(ow[i]) for i in range(0, 8, 2)]
    game.final_points = [float(ow[i]) for i in range(1, 8, 2)]


# ---------------------------------------------------------------------------
# Validation: replay the bookkeeping and compare with the log
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """The outcome of :func:`validate_game`."""

    n_rounds: int = 0
    n_decisions: int = 0
    errors: List[str] = field(default_factory=list)
    final_scores: Optional[List[int]] = None
    reached_game_end: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        head = "OK" if self.ok else f"FAILED ({len(self.errors)} error(s))"
        lines = [
            f"mjlog validation: {head}",
            f"  rounds:    {self.n_rounds}",
            f"  decisions: {self.n_decisions}",
        ]
        if self.final_scores is not None:
            lines.append("  final:     " + ", ".join(str(s * 100) for s in self.final_scores))
        for err in self.errors:
            lines.append(f"  - {err}")
        return "\n".join(lines)


def validate_game(game: MjlogGame, raise_on_error: bool = True) -> ValidationReport:
    """Re-derive every score / kyotaku / honba / hand transition.

    The running score vector is recomputed independently from the recorded
    deltas and is checked against every *authoritative* score the log states
    (``INIT`` opening scores, ``REACH`` confirmations, ``AGARI`` / ``RYUUKYOKU``
    ``sc`` snapshots and the closing ``owari``).  Riichi sticks on the table
    (kyotaku) and per-round tile flow are validated as well.  Total points are
    conserved at 100000 throughout.
    """
    report = ValidationReport(n_rounds=len(game.rounds))

    def err(msg: str) -> None:
        report.errors.append(msg)
        if raise_on_error:
            raise MjlogValidationError(msg)

    score: Optional[List[int]] = None
    kyotaku = 0  # riichi sticks on the table, in units of 100 (10 == 1000)

    for r in game.rounds:
        report.n_decisions += len(r.decisions)
        label = f"round#{r.round}.{r.honba}"

        # 1. opening scores must continue from the previous round.
        if score is not None and r.init_scores != score:
            err(f"{label}: INIT ten {r.init_scores} != carried score {score}")
        # 2. riichi sticks carried on the table must match the INIT seed.
        if kyotaku != r.riichi_sticks * 10:
            err(f"{label}: kyotaku {kyotaku} != INIT riichi sticks " f"{r.riichi_sticks} (*10)")
        score = list(r.init_scores)

        # 3. tile flow: every discard came from a hand, calls consume hand
        #    tiles, no duplicate tile ids.
        _validate_round_flow(r, err, label)

        # 4. riichi confirmations: each places a 1000-point stick. Cross-check
        #    the score snapshot Tenhou records at ``<REACH step="2">``.
        for who, ten in r.riichi_confirmations:
            score[who] -= 10
            if ten and ten != score:
                err(f"{label}: REACH(who={who}) ten {ten} != running {score}")
            kyotaku += 10

        # 5. apply the terminal deltas and reconcile the snapshot.
        for result in r.results:
            if result.score_before != score:
                err(
                    f"{label}: result sc-before {result.score_before} != " f"running score {score}"
                )
            delta_sum = sum(result.score_delta)
            if isinstance(result, Win):
                if delta_sum != kyotaku:
                    err(f"{label}: AGARI delta sum {delta_sum} != " f"kyotaku {kyotaku}")
                kyotaku = 0  # winner collects the sticks
            else:
                if delta_sum != 0:
                    err(f"{label}: RYUUKYOKU delta sum {delta_sum} != 0")
                # exhaustive/abortive draw: sticks stay on the table.
            score = [score[i] + result.score_delta[i] for i in range(4)]

    # 6. reconcile the closing scores.
    if game.final_scores is not None and score is not None:
        report.reached_game_end = True
        if kyotaku:  # leftover sticks go to the leader (lowest seat on a tie).
            lead = max(range(4), key=lambda i: (score[i], -i))
            score[lead] += kyotaku
            kyotaku = 0
        if score != game.final_scores:
            err(f"owari {game.final_scores} != reconciled {score}")
        report.final_scores = game.final_scores

    # 7. total points are always conserved.
    if score is not None:
        total = sum(score) * 100 + kyotaku * 100
        if total != 100000:
            err(f"point conservation broken: total = {total} (expected 100000)")

    return report


def _validate_round_flow(r: RoundLog, err, label: str) -> None:
    """Replay one round's tile flow.

    Validates that opening hands are 13 tiles with no duplicate ids and that
    every discard came from a tile the player actually held (calls consume the
    appropriate tiles from the caller's concealed hand).
    """
    # Every tile id 0..135 is unique within a round, so a set models the
    # concealed hand exactly.
    hands = [set(h) for h in r.init_hands]
    seen = set()
    for i, h in enumerate(r.init_hands):
        if len(h) != 13:
            err(f"{label}: player {i} opening hand has {len(h)} tiles")
        dup = seen & set(h)
        if dup:
            err(f"{label}: duplicate tile id(s) {sorted(dup)} at INIT")
        seen |= set(h)

    for e in r.events:
        if e.type is DecisionType.DRAW:
            if e.tile in seen:
                err(f"{label}: tile {e.tile} drawn twice in the round")
            seen.add(e.tile)
            hands[e.who].add(e.tile)
        elif e.type in (
            DecisionType.DISCARD,
            DecisionType.TSUMOGIRI,
            DecisionType.RIICHI,
        ):
            if e.tile not in hands[e.who]:
                err(f"{label}: player {e.who} discarded {e.tile} not in hand")
            hands[e.who].discard(e.tile)
        elif e.meld is not None:
            meld = e.meld
            if meld.kind == "closed_kan":
                from_hand = list(meld.tiles)
            elif meld.kind == "added_kan":
                from_hand = [meld.tiles[3]]  # only the added tile leaves hand
            else:  # chi / pon / open_kan: every tile except the stolen one
                from_hand = [t for i, t in enumerate(meld.tiles) if i != meld.called]
            for t in from_hand:
                if t not in hands[meld.who]:
                    err(f"{label}: player {meld.who} {meld.kind} uses {t} " f"not in hand")
                hands[meld.who].discard(t)


# ---------------------------------------------------------------------------
# MjlogReplayAgent
# ---------------------------------------------------------------------------

# The agent integrates with the native Mjx engine when it is available so that
# it can be driven through ``mjx.MjxEnv`` like any other ``mjx.Agent``.  The
# parsing / validation API above works without the engine, so we fall back to a
# plain base class when ``_mjx`` is not built.
try:  # pragma: no cover - exercised only when the native extension is built
    from mjx.action import Action as _Action  # type: ignore
    from mjx.agents import Agent as _AgentBase  # type: ignore
    from mjx.const import ActionType as _ActionType  # type: ignore

    _ENGINE_AVAILABLE = True
except Exception:  # ModuleNotFoundError when _mjx is missing, etc.
    _AgentBase = object  # type: ignore
    _Action = None  # type: ignore
    _ActionType = None  # type: ignore
    _ENGINE_AVAILABLE = False


# Maps our DecisionType onto the engine's ActionType names (resolved lazily).
_DECISION_TO_ACTION_NAME = {
    DecisionType.DISCARD: "DISCARD",
    DecisionType.TSUMOGIRI: "TSUMOGIRI",
    DecisionType.RIICHI: "RIICHI",
    DecisionType.CHI: "CHI",
    DecisionType.PON: "PON",
    DecisionType.OPEN_KAN: "OPEN_KAN",
    DecisionType.CLOSED_KAN: "CLOSED_KAN",
    DecisionType.ADDED_KAN: "ADDED_KAN",
    DecisionType.TSUMO: "TSUMO",
    DecisionType.RON: "RON",
    DecisionType.ABORTIVE_DRAW_NINE_TERMINALS: "ABORTIVE_DRAW_NINE_TERMINALS",
}


class MjlogReplayAgent(_AgentBase):  # type: ignore[misc]
    """An :class:`mjx.Agent` that replays a recorded Tenhou game from the log.

    Its only job is the agent contract: given an observation, return the action
    that was recorded in the ``.mjlog`` for that player at that point (a forced
    choice is returned directly; when the player is merely being offered a call
    they declined, a pass is returned). It plugs into the same machinery as any
    other agent -- ``act_batch`` and ``serve`` are inherited unchanged, and it
    passes ``mjx.agents.validate_agent``.

    It carries no engine or decoder logic: decoding a ``.mjlog`` into an
    ``mjx.State`` (wall reconstruction etc.) and validating the engine are
    separate concerns, not the agent's. For log-only score/hand consistency see
    :meth:`MjlogGame.validate`.

        agent = MjlogReplayAgent.from_file("game.mjlog")
        action = agent.act(observation)
    """

    def __init__(self, game: MjlogGame) -> None:
        if _ENGINE_AVAILABLE:
            _AgentBase.__init__(self)  # type: ignore[misc]
        self.game = game
        self._reset_cursor()

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "MjlogReplayAgent":
        return cls(parse_mjlog(path))

    @classmethod
    def from_str(cls, xml: str) -> "MjlogReplayAgent":
        return cls(parse_mjlog(xml))

    def decisions(self) -> List[Decision]:
        """Return every recorded decision across the whole game, in order."""
        out: List[Decision] = []
        for r in self.game.rounds:
            out.extend(r.decisions)
        return out

    def _reset_cursor(self) -> None:
        # Per-player FIFO queues of that player's positive decisions.
        self._queues: Dict[int, List[Decision]] = {i: [] for i in range(4)}
        for d in self.decisions():
            self._queues[d.who].append(d)
        self._heads: Dict[int, int] = {i: 0 for i in range(4)}
        # A riichi is logged as one decision but is two actions in mjx (declare,
        # then discard the riichi tile). After we emit the declaration this holds
        # the tile the same player must discard next.
        self._pending_riichi_discard: Dict[int, Optional[int]] = {
            i: None for i in range(4)
        }

    # -- mjx.Agent interface (requires the native engine) -------------------
    def act(self, observation):  # type: ignore[override]
        """Return the recorded action matching ``observation``.

        Requires the native ``_mjx`` extension.  The player's next recorded
        positive decision is matched against the legal actions and consumed;
        when nothing matches, a single forced action is taken as-is and an
        offered-but-declined call becomes a pass/no-op.
        """
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(
                "MjlogReplayAgent.act requires the native Mjx engine (_mjx). "
                "Use validate()/decisions() for engine-independent replay."
            )

        legal = observation.legal_actions()
        who = observation.who()

        # Second half of a riichi: discard the declared tile. This belongs to the
        # already-consumed riichi decision, so it does not advance the queue.
        pending = self._pending_riichi_discard[who]
        if pending is not None:
            match = self._match_tile(
                legal, pending, (_ActionType.DISCARD, _ActionType.TSUMOGIRI)
            )
            if match is not None:
                self._pending_riichi_discard[who] = None
                return match

        head = self._heads[who]
        queue = self._queues[who]
        if head < len(queue):
            want = queue[head]
            match = self._match(want, legal)
            if match is not None:
                self._heads[who] = head + 1
                if want.type is DecisionType.RIICHI:
                    self._pending_riichi_discard[who] = want.tile
                return match

        # No recorded positive decision applies here: either an env-forced step
        # (a single legal action, e.g. a forced tsumogiri or a round-terminal
        # dummy) or a call the player declined (pass / no-op).
        if len(legal) == 1:
            return legal[0]
        for action in legal:
            if action.type() == _ActionType.PASS:
                return action
        return legal[0]

    @staticmethod
    def _match(decision: Decision, legal):
        """Return the legal action realizing ``decision``, or ``None``.

        Matching is strict: a decision is consumed only when an offered action
        unambiguously realizes it (a meld's exact ``m`` bit field, or a tile's
        exact id / type). This is what keeps replay in lock-step -- a loose match
        could consume a *future* decision against an offered-but-declined call
        and desync every later turn.
        """
        target = getattr(_ActionType, _DECISION_TO_ACTION_NAME[decision.type])
        candidates = [a for a in legal if a.type() == target]
        if not candidates:
            return None
        # Riichi declaration is a single tile-less action (the riichi discard is
        # emitted separately; see act()).
        if decision.type is DecisionType.RIICHI:
            return candidates[0]
        # Melds: mjx's Open and Tenhou's ``m`` agree on which tile *types* form
        # the call and on the direction it was called from, but may encode
        # different *copies* of a tile (e.g. which 3p, or red vs normal five).
        # So match on (call direction, tile-type multiset) and break ties by
        # exact-id overlap. The direction matters: the same honor pon can be
        # offered on two different players' discards of that honor, and only one
        # is the recorded call.
        if decision.meld is not None:
            want_from = decision.meld.m & 0x3  # 0=self,1=right,2=across,3=left
            want_types = sorted(tile_type(t) for t in decision.meld.tiles)
            want_ids = set(decision.meld.tiles)
            best, best_overlap = None, -1
            for a in candidates:
                o = a.open()
                if o is None:
                    continue
                if int(o.steal_from()) != want_from:
                    continue
                tiles = o.tiles()
                if sorted(t.type() for t in tiles) != want_types:
                    continue
                overlap = len({t.id() for t in tiles} & want_ids)
                if overlap > best_overlap:
                    best, best_overlap = a, overlap
            return best
        # Tile-bearing decisions (discard / tsumogiri / tsumo / ron): match the
        # exact tile id (distinguishes red fives), then the tile type. Never
        # match loosely -- if nothing fits, this offer is not this decision.
        if decision.tile is not None:
            for a in candidates:
                tl = a.tile()
                if tl is not None and tl.id() == decision.tile:
                    return a
            want_type = tile_type(decision.tile)
            for a in candidates:
                tl = a.tile()
                if tl is not None and tl.type() == want_type:
                    return a
            return None
        return candidates[0]

    @staticmethod
    def _match_tile(legal, tile_id: int, types):
        candidates = [a for a in legal if a.type() in types]
        for a in candidates:
            tl = a.tile()
            if tl is not None and tl.id() == tile_id:
                return a
        want_type = tile_type(tile_id)
        for a in candidates:
            tl = a.tile()
            if tl is not None and tl.type() == want_type:
                return a
        return None


# ---------------------------------------------------------------------------
# Engine replay validation (驗引擎): drive the native Mjx engine with the
# reproduced wall + recorded actions and check the *engine's* state computation
# against the log.  This is distinct from validate_game(), which only checks the
# log's own self-consistency without an engine.
# ---------------------------------------------------------------------------


@dataclass
class EngineRoundResult:
    """Per-round outcome of comparing the engine to the log."""

    round: int
    honba: int
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class EngineReplayReport:
    """The outcome of :func:`replay_in_engine`."""

    n_rounds_log: int = 0
    n_rounds_engine: int = 0
    rounds: List[EngineRoundResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)  # game-level errors
    engine_final_scores: Optional[List[int]] = None  # units of 100

    @property
    def ok(self) -> bool:
        return not self.errors and all(r.ok for r in self.rounds)

    def summary(self) -> str:
        head = "OK" if self.ok else f"FAILED ({self._n_errors()} error(s))"
        lines = [
            f"engine replay validation: {head}",
            f"  rounds (log/engine): {self.n_rounds_log} / {self.n_rounds_engine}",
        ]
        if self.engine_final_scores is not None:
            lines.append(
                "  engine final: "
                + ", ".join(str(s * 100) for s in self.engine_final_scores)
            )
        for err in self.errors:
            lines.append(f"  - {err}")
        for r in self.rounds:
            for err in r.errors:
                lines.append(f"  - round#{r.round}.{r.honba}: {err}")
        return "\n".join(lines)

    def _n_errors(self) -> int:
        return len(self.errors) + sum(len(r.errors) for r in self.rounds)


def _tens(score: Dict) -> List[int]:
    """Length-4 ``tens`` from a protobuf-json ``Score`` (defaults omit zeros)."""
    t = list(score.get("tens", [])) if score else []
    return (t + [0, 0, 0, 0])[:4]


def _ten_changes(obj: Dict) -> List[int]:
    t = list(obj.get("tenChanges", [])) if obj else []
    return (t + [0, 0, 0, 0])[:4]


def replay_in_engine(
    game: MjlogGame, raise_on_error: bool = False
) -> EngineReplayReport:
    """Replay ``game`` through :class:`mjx.MjxEnv` and validate the engine.

    Four :class:`MjlogReplayAgent` instances (one per seat) play the recorded
    actions against each other in the engine.  The wall is reproduced from the
    game's ``SHUFFLE`` seed (:meth:`MjxEnv.reset_from_tenhou_seed`) so the engine
    deals exactly the tiles Tenhou dealt; the engine then computes every state
    transition (draws, melds, riichi, yaku/fu/score, kyotaku, honba) on its own.
    Each round's engine-computed terminal (final score, win deltas, fu, ten,
    tenpai payments) is compared against the values recorded in the log.

    Unlike :func:`validate_game`, this exercises the *engine*: a mismatch points
    at the engine's state computation, not at the log.
    """
    if not _ENGINE_AVAILABLE:
        raise RuntimeError(
            "replay_in_engine requires the native Mjx engine (_mjx). "
            "Use validate_game() for engine-independent log validation."
        )
    if not game.seed:
        raise MjlogValidationError(
            "mjlog has no <SHUFFLE> seed; cannot reproduce the wall"
        )

    import json

    from mjx.env import MjxEnv

    report = EngineReplayReport(n_rounds_log=len(game.rounds))

    def gerr(msg: str) -> None:
        report.errors.append(msg)
        if raise_on_error:
            raise MjlogValidationError(msg)

    # Seat i (mjlog ``hai{i}`` / dealer of round 0) maps to engine AbsolutePos i.
    player_ids = ["0", "1", "2", "3"]
    agents = {pid: MjlogReplayAgent(game) for pid in player_ids}
    env = MjxEnv(player_ids)
    obs = env.reset_from_tenhou_seed(game.seed, player_ids)

    captured: List[Dict] = []
    prev_round_over = False
    # The recorded game is finite; the bound only guards against a divergence
    # that would otherwise loop forever.
    for _ in range(1_000_000):
        if env.done("game"):
            break
        action_dict = {pid: agents[pid].act(o) for pid, o in obs.items()}
        obs = env.step(action_dict)
        round_over = env.done("round")
        if round_over and not prev_round_over:
            captured.append(json.loads(env.state().to_json()))
        prev_round_over = round_over
    else:  # pragma: no cover - divergence guard
        gerr("engine replay did not terminate (diverged from the log?)")

    report.n_rounds_engine = len(captured)
    if captured:
        last = captured[-1].get("roundTerminal", {}).get("finalScore", {})
        report.engine_final_scores = [t // 100 for t in _tens(last)]

    # A complete game (one that records its closing ``owari``) must reproduce
    # exactly. Hand-crafted partial logs simply stop early, so for those we only
    # validate the rounds the log actually contains.
    if game.final_scores is not None and len(captured) != len(game.rounds):
        gerr(
            f"round count mismatch: log has {len(game.rounds)}, "
            f"engine produced {len(captured)}"
        )

    for i, mlog in enumerate(game.rounds):
        rr = EngineRoundResult(round=mlog.round, honba=mlog.honba)
        report.rounds.append(rr)
        if i >= len(captured):
            rr.errors.append("engine produced no terminal for this round")
            continue

        def rerr(msg: str) -> None:
            rr.errors.append(msg)
            if raise_on_error:
                raise MjlogValidationError(f"round#{mlog.round}.{mlog.honba}: {msg}")

        state = captured[i]
        init = state.get("publicObservation", {}).get("initScore", {})
        term = state.get("roundTerminal", {})

        # 1. opening situation the engine reached for this round.
        if init.get("round", 0) != mlog.round or init.get("honba", 0) != mlog.honba:
            rerr(
                f"engine reached round {init.get('round', 0)}."
                f"{init.get('honba', 0)} != log {mlog.round}.{mlog.honba}"
            )
        if init.get("riichi", 0) != mlog.riichi_sticks:
            rerr(
                f"init riichi sticks {init.get('riichi', 0)} != "
                f"log {mlog.riichi_sticks}"
            )
        exp_init = [s * 100 for s in mlog.init_scores]
        if _tens(init) != exp_init:
            rerr(f"init tens {_tens(init)} != log {exp_init}")

        # 2. result type + per-result figures.
        log_wins = [r for r in mlog.results if isinstance(r, Win)]
        log_draws = [r for r in mlog.results if isinstance(r, ExhaustiveDraw)]
        eng_wins = term.get("wins", [])
        eng_no_winner = term.get("noWinner")

        if log_wins:
            if not eng_wins:
                rerr("log records a win but the engine has no winner")
            for mw in log_wins:
                ew = next((w for w in eng_wins if w.get("who", 0) == mw.who), None)
                if ew is None:
                    rerr(f"engine missing win for player {mw.who}")
                    continue
                if ew.get("fromWho", 0) != mw.from_who:
                    rerr(
                        f"win(who={mw.who}) fromWho {ew.get('fromWho', 0)} != "
                        f"log {mw.from_who}"
                    )
                # mjx reports fu = 0 for limit hands (mangan and above), where
                # fu does not determine the score; Tenhou still records it
                # cosmetically. Only compare fu for non-limit hands.
                if mw.limit == 0 and ew.get("fu", 0) != mw.fu:
                    rerr(f"win(who={mw.who}) fu {ew.get('fu', 0)} != log {mw.fu}")
                if ew.get("ten", 0) != mw.points:
                    rerr(
                        f"win(who={mw.who}) ten {ew.get('ten', 0)} != "
                        f"log {mw.points}"
                    )
                exp_delta = [d * 100 for d in mw.score_delta]
                if _ten_changes(ew) != exp_delta:
                    rerr(
                        f"win(who={mw.who}) ten_changes {_ten_changes(ew)} != "
                        f"log {exp_delta}"
                    )
        elif log_draws:
            if eng_no_winner is None:
                rerr("log records an exhaustive/abortive draw but engine has a winner")
            else:
                md = log_draws[0]
                exp_delta = [d * 100 for d in md.score_delta]
                if _ten_changes(eng_no_winner) != exp_delta:
                    rerr(
                        f"draw ten_changes {_ten_changes(eng_no_winner)} != "
                        f"log {exp_delta}"
                    )
                log_tenpai = {i for i, t in enumerate(md.tenpai) if t}
                eng_tenpai = {
                    t.get("who", 0) for t in eng_no_winner.get("tenpais", [])
                }
                if eng_tenpai != log_tenpai:
                    rerr(
                        f"tenpai players {sorted(eng_tenpai)} != "
                        f"log {sorted(log_tenpai)}"
                    )

        # 3. running score the engine carried out of the round.
        final = term.get("finalScore", {})
        if i + 1 < len(game.rounds):
            exp_final = [s * 100 for s in game.rounds[i + 1].init_scores]
        elif game.final_scores is not None:
            exp_final = [s * 100 for s in game.final_scores]
        else:
            exp_final = None
        if exp_final is not None and _tens(final) != exp_final:
            rerr(f"final tens {_tens(final)} != log {exp_final}")

    return report


def replay_and_validate(source: str, raise_on_error: bool = True) -> ValidationReport:
    """Convenience: parse ``source`` (path or XML) and validate it."""
    return parse_mjlog(source).validate(raise_on_error=raise_on_error)


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import glob as _glob

    parser = argparse.ArgumentParser(
        description="Replay Tenhou mjlog files and validate state computation."
    )
    parser.add_argument("paths", nargs="+", help="mjlog file(s) or glob(s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print a per-file report")
    args = parser.parse_args(argv)

    files: List[str] = []
    for p in args.paths:
        files.extend(sorted(_glob.glob(p)) or [p])

    n_ok = 0
    for path in files:
        report = parse_mjlog(path).validate(raise_on_error=False)
        if report.ok:
            n_ok += 1
        name = path.rsplit("/", 1)[-1]
        status = "OK  " if report.ok else "FAIL"
        print(f"{status} {name}  rounds={report.n_rounds}")
        if args.verbose or not report.ok:
            print(report.summary())
    print(f"\n{n_ok}/{len(files)} files validated")
    return 0 if n_ok == len(files) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
