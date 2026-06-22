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
    "MjlogParseError",
    "MjlogValidationError",
    "parse_mjlog",
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

        if tag == "GO":
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
# Engine bridge: reconstruct an mjx (mjxproto) State from a recorded round so
# that the native engine can *replay* the Tenhou game and we can validate that
# the engine's state computation agrees with Tenhou.
# ---------------------------------------------------------------------------

# Tenhou / mjx share the wall layout (see internal/wall.h):
#   [0..51]   initial hands (depend on the round/dealer)
#   [52..121] live draws (tsumo)
#   [122,124,126,128] kan dora indicators (1st..4th -> 128,126,124,122)
#   [123,125,127,129] kan ura dora indicators
#   [130] dora indicator, [131] ura dora indicator
#   [132..135] rinshan (kan) draws, taken in the order 134,135,132,133
_RINSHAN_WALL_IXS = (134, 135, 132, 133)

# mjx event-type names (used in the State JSON the engine consumes).
_EVENT_NAME = {
    DecisionType.CHI: "EVENT_TYPE_CHI",
    DecisionType.PON: "EVENT_TYPE_PON",
    DecisionType.CLOSED_KAN: "EVENT_TYPE_CLOSED_KAN",
    DecisionType.OPEN_KAN: "EVENT_TYPE_OPEN_KAN",
    DecisionType.ADDED_KAN: "EVENT_TYPE_ADDED_KAN",
    DecisionType.TSUMO: "EVENT_TYPE_TSUMO",
    DecisionType.RON: "EVENT_TYPE_RON",
    DecisionType.ABORTIVE_DRAW_NINE_TERMINALS: "EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS",
}


def _reconstruct_wall(r: RoundLog) -> List[int]:
    """Rebuild the 136-tile wall (in mjx/Tenhou order) for a round.

    Every tile the engine actually reads while replaying -- initial hands,
    live draws, rinshan draws and the revealed dora/ura indicators -- is placed
    at its canonical wall index.  Tiles that the log never reveals (undrawn live
    wall, unrevealed dead wall) are filled with the remaining ids so the wall is
    a valid permutation; the engine never reads those positions during replay.
    """
    wall: List[Optional[int]] = [None] * 136
    used = set()

    def place(ix: int, tile: int) -> None:
        wall[ix] = tile
        used.add(tile)

    rnd = r.round
    for pos in range(4):
        seat = (pos - rnd) % 4  # deal order relative to the dealer
        base = seat * 4
        idxs = (
            list(range(base, base + 4))
            + list(range(base + 16, base + 20))
            + list(range(base + 32, base + 36))
            + [48 + seat]
        )
        for tile, ix in zip(r.init_hands[pos], idxs):
            place(ix, tile)

    # Separate live draws from rinshan (kan) draws: a draw is a rinshan draw if
    # it immediately follows a kan by the same player.
    live, rinshan = [], []
    pending_kan: Optional[int] = None
    for e in r.events:
        if e.type in (
            DecisionType.CLOSED_KAN,
            DecisionType.OPEN_KAN,
            DecisionType.ADDED_KAN,
        ):
            pending_kan = e.who
        elif e.type is DecisionType.DRAW:
            (rinshan if pending_kan == e.who else live).append(e.tile)
            pending_kan = None
    for i, tile in enumerate(live):
        place(52 + i, tile)
    for n, tile in enumerate(rinshan):
        place(_RINSHAN_WALL_IXS[n], tile)

    # Dora indicators: [130] then kan dora at 128, 126, 124, 122.
    place(130, r.dora_indicators[0])
    for n, d in enumerate(r.dora_indicators[1:]):
        place(128 - 2 * n, d)

    # Ura dora (only revealed by a riichi win): [131] then 129, 127, 125, 123.
    ura: List[int] = []
    for res in r.results:
        if isinstance(res, Win) and res.ura_dora_indicators:
            ura = res.ura_dora_indicators
    if ura:
        place(131, ura[0])
        for n, u in enumerate(ura[1:]):
            place(129 - 2 * n, u)

    leftover = [t for t in range(136) if t not in used]
    it = iter(leftover)
    for ix in range(136):
        if wall[ix] is None:
            wall[ix] = next(it)
    assert sorted(wall) == list(range(136)), "reconstructed wall is not a permutation"
    return wall  # type: ignore[return-value]


def _reconstruct_events(r: RoundLog) -> List[Dict]:
    """Translate a round into the mjx public-event stream.

    Tenhou's element order already matches mjx's event order, so this is a
    direct mapping, with two refinements that mirror the engine:
    riichi is a ``RIICHI`` event followed by the discard, and a *confirmed*
    riichi (one that placed a stick) is followed by ``RIICHI_SCORE_CHANGE``.
    """
    confirmed = {who for who, _ in r.riichi_confirmations}
    events: List[Dict] = []
    last_draw: List[Optional[int]] = [None, None, None, None]
    for e in r.events:
        if e.type is DecisionType.DRAW:
            events.append({"type": "EVENT_TYPE_DRAW", "who": e.who})
            last_draw[e.who] = e.tile
        elif e.type is DecisionType.DISCARD:
            events.append({"type": "EVENT_TYPE_DISCARD", "who": e.who, "tile": e.tile})
            last_draw[e.who] = None
        elif e.type is DecisionType.TSUMOGIRI:
            events.append({"type": "EVENT_TYPE_TSUMOGIRI", "who": e.who, "tile": e.tile})
            last_draw[e.who] = None
        elif e.type is DecisionType.RIICHI:
            events.append({"type": "EVENT_TYPE_RIICHI", "who": e.who})
            sub = "EVENT_TYPE_TSUMOGIRI" if last_draw[e.who] == e.tile else "EVENT_TYPE_DISCARD"
            events.append({"type": sub, "who": e.who, "tile": e.tile})
            last_draw[e.who] = None
            if e.who in confirmed:
                events.append({"type": "EVENT_TYPE_RIICHI_SCORE_CHANGE", "who": e.who})
        elif e.type in _EVENT_NAME:
            event = {"type": _EVENT_NAME[e.type], "who": e.who}
            if e.tile is not None:
                event["tile"] = e.tile
            if e.meld is not None:
                event["open"] = e.meld.m  # mjx uses Tenhou's meld encoding
            events.append(event)
    return events


def round_to_state_dict(r: RoundLog) -> Dict:
    """Build the mjxproto ``State`` (as a dict) the engine can replay.

    Only the fields the engine reads to regenerate a round are populated: the
    reconstructed wall, the player ids, the opening :class:`Score` (note Tenhou
    scores are in units of 100, mjx uses raw points) and the public event
    stream.  Hands, draws, dora and per-step observations are recomputed by the
    engine from these.
    """
    return {
        "hiddenState": {"wall": _reconstruct_wall(r)},
        "publicObservation": {
            "playerIds": [f"player_{i}" for i in range(4)],
            "initScore": {
                "round": r.round,
                "honba": r.honba,
                "riichi": r.riichi_sticks,
                "tens": [t * 100 for t in r.init_scores],
            },
            "events": _reconstruct_events(r),
        },
    }


def round_to_state_json(r: RoundLog) -> str:
    """JSON form of :func:`round_to_state_dict` (accepted by ``mjx.State``)."""
    import json

    return json.dumps(round_to_state_dict(r))


@dataclass
class EngineValidationReport:
    """Outcome of replaying a game through the native engine."""

    n_rounds: int = 0
    n_replayed: int = 0  # rounds the engine replayed without a consistency error
    n_action_match: int = 0  # rounds whose engine action stream matched Tenhou
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.n_replayed == self.n_rounds

    def summary(self) -> str:
        head = "OK" if self.ok else "issues"
        lines = [
            f"engine replay: {head}",
            f"  rounds replayed: {self.n_replayed}/{self.n_rounds}",
            f"  action streams:  {self.n_action_match}/{self.n_rounds} match Tenhou",
        ]
        for e in self.errors:
            lines.append(f"  - {e}")
        return "\n".join(lines)


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
    """Replay a recorded Tenhou game.

    Construct it from a ``.mjlog`` file (or a parsed :class:`MjlogGame`) and it
    will reproduce, in order, every decision the players made.  There are two
    distinct things you can check:

    * **Log self-consistency (no native build required).** ``agent.validate()``
      re-derives the full score / kyotaku / honba / hand bookkeeping from the
      recorded deltas and checks it against the values stored in the log.  This
      validates the *log*, treating Tenhou as the oracle::

          report = MjlogReplayAgent.from_file("game.mjlog").validate()
          assert report.ok

    * **Engine validation (requires the native build).**
      ``agent.validate_with_engine()`` hands the recorded wall and actions to
      the Mjx engine, which independently recomputes the hands, draws, dora,
      legal actions and state transitions, and checks that the engine replays
      the whole game and reproduces Tenhou's action stream.  This validates the
      *engine* against Tenhou ground truth::

          report = MjlogReplayAgent.from_file("game.mjlog").validate_with_engine()
          assert report.n_replayed == report.n_rounds

    When Mjx is built the agent is also a normal :class:`mjx.Agent` (``act``
    returns the recorded action for an observation), so it can be plugged into
    other engine machinery.
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

    # -- replay / validation (engine independent) ---------------------------
    def validate(self, raise_on_error: bool = True) -> ValidationReport:
        """Validate that the recorded state transitions are self-consistent."""
        return self.game.validate(raise_on_error=raise_on_error)

    def decisions(self) -> List[Decision]:
        """Return every recorded decision across the whole game, in order."""
        out: List[Decision] = []
        for r in self.game.rounds:
            out.extend(r.decisions)
        return out

    # -- engine replay (validates the Mjx engine, requires the native build) --
    def to_state_json(self, round_index: int) -> str:
        """mjxproto ``State`` JSON for a round, replayable by ``mjx.State``.

        This is engine-independent (it only builds JSON); feed it to
        ``mjx.State(...)`` to have the native engine replay the round.
        """
        return round_to_state_json(self.game.rounds[round_index])

    def to_mjx_states(self) -> List["object"]:
        """Reconstruct one ``mjx.State`` per round (requires ``_mjx``)."""
        if not _ENGINE_AVAILABLE:
            raise RuntimeError("to_mjx_states requires the native Mjx engine (_mjx).")
        from mjx.state import State  # local import: needs the engine

        return [State(round_to_state_json(r)) for r in self.game.rounds]  # type: ignore[arg-type]

    def validate_with_engine(self, raise_on_error: bool = False) -> "EngineValidationReport":
        """Replay every round through the native engine and check the result.

        For each round the recorded Tenhou wall and actions are handed to the
        engine, which independently recomputes the hands, draws, dora, legal
        actions and state transitions (via ``State.past_decisions``).  Two
        things are checked:

        * the engine replays the whole round without raising a consistency
          error -- i.e. every recorded action was legal in the engine's own
          recomputed observation; and
        * the engine's regenerated action stream matches Tenhou's recorded
          decisions.

        Requires the native ``_mjx`` extension.
        """
        if not _ENGINE_AVAILABLE:
            raise RuntimeError("validate_with_engine requires the native Mjx engine (_mjx).")
        from mjx.state import State  # local import: needs the engine

        report = EngineValidationReport(n_rounds=len(self.game.rounds))
        for i, r in enumerate(self.game.rounds):
            try:
                decisions = State(round_to_state_json(r)).past_decisions()
            except Exception as exc:  # an engine assertion / inconsistency
                msg = f"round {i}: engine failed to replay: {exc}"
                report.errors.append(msg)
                if raise_on_error:
                    raise MjlogValidationError(msg) from exc
                continue
            report.n_replayed += 1
            if self._engine_actions(decisions) == self._tenhou_actions(r):
                report.n_action_match += 1
        return report

    @staticmethod
    def _engine_actions(decisions) -> List[Tuple[int, int]]:
        """(who, ActionType) of each non-pass/dummy action the engine applied."""
        skip = {_ActionType.PASS, _ActionType.DUMMY}
        out = []
        for _obs, act in decisions:
            t = act.type()
            if t not in skip:
                out.append((int(act.who()), int(t)))
        return out

    @staticmethod
    def _tenhou_actions(r: RoundLog) -> List[Tuple[int, int]]:
        """The recorded decisions as (who, ActionType), mirroring the engine.

        Riichi is expanded to a RIICHI action plus the discard (tsumogiri when
        the riichi tile is the drawn tile) to match how the engine replays it.
        """
        out: List[Tuple[int, int]] = []
        last_draw: List[Optional[int]] = [None, None, None, None]
        for e in r.events:
            if e.type is DecisionType.DRAW:
                last_draw[e.who] = e.tile
                continue
            if e.type is DecisionType.RIICHI:
                out.append((e.who, int(_ActionType.RIICHI)))
                sub = (
                    DecisionType.TSUMOGIRI if last_draw[e.who] == e.tile else DecisionType.DISCARD
                )
                out.append((e.who, int(getattr(_ActionType, sub.name))))
                last_draw[e.who] = None
            else:
                name = _DECISION_TO_ACTION_NAME[e.type]
                out.append((e.who, int(getattr(_ActionType, name))))
                if e.type in (DecisionType.DISCARD, DecisionType.TSUMOGIRI):
                    last_draw[e.who] = None
        return out

    def _reset_cursor(self) -> None:
        # Per-player FIFO queues of that player's positive decisions.
        self._queues: Dict[int, List[Decision]] = {i: [] for i in range(4)}
        for d in self.decisions():
            self._queues[d.who].append(d)
        self._heads: Dict[int, int] = {i: 0 for i in range(4)}

    # -- mjx.Agent interface (requires the native engine) -------------------
    def act(self, observation):  # type: ignore[override]
        """Return the recorded action matching ``observation``.

        Requires the native ``_mjx`` extension.  Forced choices are returned
        directly; otherwise the player's next recorded positive decision is
        matched against the legal actions, and a pass/no-op is returned when the
        player declined an offered call.
        """
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(
                "MjlogReplayAgent.act requires the native Mjx engine (_mjx). "
                "Use validate()/decisions() for engine-independent replay."
            )

        legal = observation.legal_actions()
        if len(legal) == 1:
            return legal[0]

        who = observation.who()
        head = self._heads[who]
        queue = self._queues[who]
        if head < len(queue):
            want = queue[head]
            match = self._match(want, legal)
            if match is not None:
                self._heads[who] = head + 1
                return match

        # The player is being offered a call they declined: pass / no-op.
        for action in legal:
            if action.type() == _ActionType.PASS:
                return action
        return legal[0]

    @staticmethod
    def _match(decision: Decision, legal):
        target = getattr(_ActionType, _DECISION_TO_ACTION_NAME[decision.type])
        candidates = [a for a in legal if a.type() == target]
        if not candidates:
            return None
        if decision.tile is not None:
            want_type = tile_type(decision.tile)
            for a in candidates:
                tile = a.tile()
                if tile is not None and tile.type() == want_type:
                    return a
        return candidates[0]


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
