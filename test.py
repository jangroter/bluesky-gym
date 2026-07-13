"""
This file contains the example code used during the first part of the workshop.
Do not run the code directly from here, but instead, copy it from this file 
to the corresponding file as indicated in the workshop.
"""

import gymnasium as gym
from stable_baselines3 import PPO
import bluesky_gym
import bluesky_gym.envs
from bluesky_gym.utils import logger
bluesky_gym.register_envs()

# Initialize the environment and logger
env_name = 'CompetitionEnv-v0'
env = gym.make(env_name, render_mode=None)
file_name = 'my_first_bsg_experiment.csv'
logger = logger.CSVLoggerCallback('logs/', file_name)

# Train a model for 'n' timesteps
# model = PPO('MultiInputPolicy', env=env, verbose=1)
# model.learn(total_timesteps=1_00, callback=logger)
# model.save("models/SAC")
# env.close()

# model = PPO.load("models/SAC", env=env)
env = gym.make(env_name, render_mode="human")

n_eps = 10
for i in range(n_eps):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        # action, _states = model.predict(obs, deterministic=True)
        action = [0.001,0.001]
        obs, reward, done, truncated, info = env.step(action)
env.close()




