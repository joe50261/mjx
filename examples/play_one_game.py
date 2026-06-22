"""Play one full game (one 半荘 / hanchan) with RandomAgent and print results.

This is the minimal "run a game end-to-end" example from the README,
wrapped in a script so it can be executed directly:

    python3 examples/play_one_game.py
"""

import mjx
from mjx.agents import RandomAgent


def play_one_game(seed: int = 1234) -> None:
    agent = RandomAgent()
    env = mjx.MjxEnv()
    obs_dict = env.reset(seed=seed)

    steps = 0
    while not env.done():
        actions = {player_id: agent.act(obs) for player_id, obs in obs_dict.items()}
        obs_dict = env.step(actions)
        steps += 1

    returns = env.rewards()  # Tenhou 7-dan placement points (sum to 0)
    print(f"Game finished in {steps} steps (seed={seed}).")
    for player_id in sorted(returns):
        print(f"  {player_id}: {returns[player_id]:+d}")


if __name__ == "__main__":
    play_one_game()
