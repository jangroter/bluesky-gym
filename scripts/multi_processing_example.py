"""
This file is an example train and test loop for the different environments that
uses multiprocessing through the use of vectorised environments.

Note that multiprocessing doesn't necessarily result in faster training. It is
highly dependent on the environment and algorithm combination. If the algorithm
is able to train over a batch of observations, multiprocessing should lead to
faster training.

IMPORTANT: SubprocVecEnv is required here. BlueSky keeps its simulation state
in per-process module singletons (bs.sim, bs.traf, ...), so each parallel
environment must live in its own process. Running multiple environments in a
single process (e.g. DummyVecEnv with n_envs > 1) makes them share one BlueSky
instance, corrupting each other's traffic state.

Selecting different environments is done through setting the 'env_name' variable.
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

bluesky_gym.register_envs()

env_name = 'SectorCREnv-v0'
algorithm = SAC
num_cpu = 2

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name,
        monitor_keys=('total_reward', 'total_intrusions'))

TRAIN = True
EVAL_EPISODES = 10

if __name__ == "__main__":
    # make_vec_env seeds each worker with seed + rank,
    # giving every environment a different scenario stream.
    env = make_vec_env(env_name,
            n_envs=num_cpu,
            seed=0,
            vec_env_cls=SubprocVecEnv)
    model = algorithm("MultiInputPolicy", env, verbose=1, learning_rate=3e-4)
    if TRAIN:
        model.learn(total_timesteps=int(2e6), callback=csv_logger_callback)
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp")
        del model
    env.close()
    del env

    # Test the trained model
    env = gym.make(env_name, render_mode="human")
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp", env=env)
    for i in range(EVAL_EPISODES):
        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            tot_rew += reward
        print(tot_rew)
    env.close()
