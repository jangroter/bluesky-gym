"""
This file trains a model using the Descent-V0 example environment
"""
import gymnasium as gym
from stable_baselines3 import SAC

import numpy as np

import bluesky_gym
import bluesky_gym.envs

bluesky_gym.register_envs()

from bluesky_gym.utils import logger

# Initialize logger
log_dir = './logs/descent_env/'
file_name = 'DescentEnv-v0_sac.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = True

if __name__ == "__main__":
    # Create the environment
    env = gym.make('DescentEnv-v0', render_mode=None)
    obs, info = env.reset()

    # Create the model
    model = SAC("MultiInputPolicy", env, verbose=1,learning_rate=1e-3)

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=2048, callback=csv_logger_callback)
        model.save("models/DescentEnv-v0_sac/model")  
    env.close()