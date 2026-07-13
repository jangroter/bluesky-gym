"""
Manual smoke test for the competition environment.

Runs a few episodes with human rendering and a neutral (fly-straight) action,
printing the objective scoring metrics from the info dict at the end of each
episode. Not an automated test — a quick visual/behavioural check.
"""

import gymnasium as gym
import bluesky_gym
import bluesky_gym.envs
bluesky_gym.register_envs()

env_name = 'CompetitionEnv-v0'
env = gym.make(env_name, render_mode="human")

n_eps = 3
for i in range(n_eps):
    terminated = truncated = False
    obs, info = env.reset(seed=i)
    while not (terminated or truncated):
        action = [0.0, 0.0]   # neutral: keep current heading and speed
        obs, reward, terminated, truncated, info = env.step(action)
    print(f"episode {i}: terminated={terminated} truncated={truncated} info={info}")
env.close()
