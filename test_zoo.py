"""
Manual smoke test for the multi-agent competition environment.

Runs a few episodes with human rendering and neutral (fly-straight) actions
for every agent, printing each agent's objective scoring metrics as it leaves
the episode. Not an automated test — a quick visual/behavioural check.
"""

import numpy as np

from bluesky_zoo.competition_v0 import CompetitionZooEnv

env = CompetitionZooEnv(render_mode="human", n_agents=4)

n_eps = 3
for i in range(n_eps):
    observations, infos = env.reset(seed=i)
    while env.agents:
        actions = {agent: np.zeros(2) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent in sorted(infos):
            if terminations[agent] or truncations[agent]:
                print(f"episode {i} {agent}: terminated={terminations[agent]} "
                      f"truncated={truncations[agent]} info={infos[agent]}")
env.close()
