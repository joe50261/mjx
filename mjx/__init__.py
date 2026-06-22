# Engine-dependent public API. These require the native ``_mjx`` extension to
# be built. When it is missing we still expose the pure-Python utilities below
# (e.g. ``mjx.mjlog``) so that recorded games can be replayed and validated
# without compiling Mjx.
try:
    from mjx.action import Action
    from mjx.agents import Agent
    from mjx.const import ActionType, EventType, TileType
    from mjx.env import MjxEnv, run
    from mjx.event import Event
    from mjx.hand import Hand
    from mjx.observation import Observation
    from mjx.open import Open
    from mjx.state import State
    from mjx.tile import Tile

    _ENGINE_AVAILABLE = True
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on the build
    if _exc.name != "_mjx":
        raise
    _ENGINE_AVAILABLE = False

# Pure-Python utilities (always importable, no native extension required).
from mjx import mjlog
from mjx.mjlog import MjlogReplayAgent

__all__ = [
    "Action",
    "Event",
    "Observation",
    "State",
    "MjxEnv",
    "Agent",
    "Open",
    "Hand",
    "Tile",
    "run",
    "ActionType",
    "EventType",
    "TileType",
    "mjlog",
    "MjlogReplayAgent",
]
