"""
This file is an example train and test loop for the different environments.
Selecting different environments is done through setting the 'env_name' variable.

TODO:
* add rgb_array rendering for the different environments to allow saving videos
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from impalaCNN import IMPALACNN, CustomCNN, ImpalaLight, ImpalaLight_v2
import numpy as np

import bluesky_gym
import bluesky_gym.envs
import ale_py

from bluesky_gym.utils import logger

gym.register_envs(ale_py)
bluesky_gym.register_envs()

env_name = 'ALE/Breakout-v5'
algorithm = PPO

policy_kwargs = dict(
    features_extractor_class=ImpalaLight_v2,
    features_extractor_kwargs=dict(features_dim=256),  # Adjust feature dimension as needed
)

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = True
EVAL_EPISODES = 10


if __name__ == "__main__":
    env = gym.make(env_name, render_mode=None)
    obs, info = env.reset()
    # Use with any algorithm
    model = algorithm("CnnPolicy", "Breakout-v4", policy_kwargs=policy_kwargs, verbose=1)

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=100000)
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model")
        del model
    env.close()
    
    # Test the trained model
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model", env=env)
    env = gym.make(env_name, render_mode="human")
    for i in range(EVAL_EPISODES):

        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
        print(tot_rew)
    env.close()