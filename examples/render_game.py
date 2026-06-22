"""Play one game and render the board as SVG (the "Beautiful visualization").

Plays a full game with RandomAgent, captures a mid-round snapshot, and saves
two SVGs:

  * board_state.svg -- full State view (all four hands visible, god's-eye)
  * board_obs.svg   -- Observation from one player's point of view
                       (opponents' hands face-down), the README's obs.png style

Usage:
    PYTHONPATH=. python3 examples/render_game.py [out_dir] [seed] [snapshot_step]

Convert an SVG to PNG with e.g. librsvg:
    rsvg-convert -w 2000 -h 2000 -b white board_state.svg -o board_state.png
"""

import os
import sys

import mjx
from mjx.agents import RandomAgent


def render_game(out_dir: str = "viz_out", seed: int = 1234, snapshot_step: int = 45) -> None:
    os.makedirs(out_dir, exist_ok=True)

    agent = RandomAgent()
    env = mjx.MjxEnv()
    obs_dict = env.reset(seed=seed)

    snap_state = None
    snap_obs = None
    step = 0
    while not env.done():
        actions = {pid: agent.act(obs) for pid, obs in obs_dict.items()}
        if snap_state is None and step == snapshot_step:
            snap_state = env.state()
            snap_obs = obs_dict[sorted(obs_dict)[0]]
        obs_dict = env.step(actions)
        step += 1

    # Fall back to the terminal state if the game ended before the snapshot step.
    if snap_state is None:
        snap_state = env.state()

    state_path = os.path.join(out_dir, "board_state.svg")
    snap_state.save_svg(state_path, view_idx=0)
    print(f"saved {state_path}")

    if snap_obs is not None:
        obs_path = os.path.join(out_dir, "board_obs.svg")
        snap_obs.save_svg(obs_path)
        print(f"saved {obs_path}")

    print(f"game length: {step} steps; final rewards: {env.rewards()}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "viz_out"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    snapshot_step = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    render_game(out_dir, seed, snapshot_step)
