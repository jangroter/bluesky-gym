import gymnasium as gym
import numpy as np

class NoisyObservationWrapper(gym.Wrapper):
    def __init__(self, env, noise_level=0.1):
        super().__init__(env)
        self.noise_level = noise_level

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        noisy_observation = self.add_noise(observation)
        return noisy_observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        noisy_observation = self.add_noise(observation)
        return noisy_observation, reward, done, truncated, info

    def add_noise(self, observation):
        # Add Gaussian noise to the observation
        noise = np.random.normal(0, self.noise_level, size=observation.shape)
        return observation + noise