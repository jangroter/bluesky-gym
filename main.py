"""
This file is an example train and test loop for the different environments.
Selecting different environments is done through setting the 'env_name' variable.

TODO:
* add rgb_array rendering for the different environments to allow saving videos
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG

import numpy as np

import bluesky_gym
import bluesky_gym.envs
from bluesky_gym.envs.common.wrappers import WindFieldWrapper

from bluesky_gym.utils import logger
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback

bluesky_gym.register_envs()

env_name = 'PathPlanningEnv-v1'
algorithm = SAC

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = True
EVAL_EPISODES = 10

net_arch = dict(pi=[256, 256, 256], qf=[256, 256, 256])  # Separate actor (`pi`) and critic (`vf`) network

policy_kwargs = dict(
    net_arch=net_arch
)

def linear_schedule(progress_remaining):
    return max(1e-3 * progress_remaining**3,3e-5)

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        # Check if the training step is a multiple of the save frequency
        if self.n_calls % self.save_freq == 0:
            # Save the model
            model_path = f"{self.save_path}/model_50north2.zip"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
        return True


if __name__ == "__main__":
    env = gym.make(env_name, render_mode=None)
    windy_env = WindFieldWrapper(env, lat=np.array([52]), lon=np.array([4]), vnorth=np.array([[50]]), veast=np.array([[0]]), alt=None)
    # windy_env = WindFieldWrapper(env, lat=np.array([52]), lon=np.array([4]), vnorth=50, veast=0, alt=None)

    save_callback = SaveModelCallback(save_freq=1000, save_path="./saved_models", verbose=1)
    # eval_callback = EvalCallback(
    #     env,                           # Environment for evaluation
    #     best_model_save_path="./best_model",  # Path to save the best model
    #     log_path="./logs",                    # Path for evaluation logs
    #     eval_freq=1000,                       # Evaluate every 1000 steps
    #     deterministic=True,
    #     render=False,
    # )

    callback = CallbackList([save_callback,csv_logger_callback])

    # obs, info = env.reset()
    model = algorithm("MultiInputPolicy", windy_env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=linear_schedule)
    if TRAIN:
        model.learn(total_timesteps=2e5, callback=callback)
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