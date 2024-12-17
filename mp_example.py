"""
This file is an example train and test loop for the different environments that
uses multiprocessing through the use of vectorised environments.
Note that multiprocessing doesn't necessarily result in faster training. It is
highly dependent on the environment and algorithm combination. If the algorithm
is able to train over a batch of observations, multiprocessing should lead to
faster training.
Selecting different environments is done through setting the 'env_name' variable.
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

bluesky_gym.register_envs()

env_name = 'PathPlanningEnv-v0'
algorithm = SAC
num_cpu = 4

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = True
EVAL_EPISODES = 10

# Initialise the environment counter
env_counter = 0

def make_env():
    """
    Utility function for multiprocessed env.
    """
    global env_counter
    env = gym.make(env_name, 
            render_mode=None)
    # Set a different seed for each created environment.
    env.reset(seed=env_counter)
    env_counter +=1 
    return env

if __name__ == "__main__":
    env = make_vec_env(make_env, 
            n_envs = num_cpu,
            vec_env_cls=SubprocVecEnv)
    model = algorithm("MultiInputPolicy", env, verbose=1,learning_rate=3e-3)
    if TRAIN:
        model.learn(total_timesteps=2e5, callback=csv_logger_callback)
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
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
        print(tot_rew)
    env.close()