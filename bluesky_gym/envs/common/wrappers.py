import gymnasium as gym
import numpy as np

import bluesky as bs
from bluesky.traffic.windsim import WindSim

class WindFieldWrapper(gym.Wrapper):
    def __init__(self, env, lat, lon, vnorth, veast, alt):
        super().__init__(env)
        self.lat = lat
        self.lon = lon
        self.vnorth = vnorth
        self.veast = veast
        self.alt = alt

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        bs.traf.wind.addpointvne(self.lat, self.lon, self.vnorth, self.veast, self.alt) 
        return observation, info

class NoisyObservationWrapper(gym.Wrapper):
    def __init__(self, env, noise_level=0.1):
        super().__init__(env)
        self.noise_level = noise_level

    def reset(self, **kwargs):
        # Reset the environment and get the initial observation
        observation, info = self.env.reset(**kwargs)
        # Add noise to the observation
        noisy_observation = self.add_noise(observation)
        return noisy_observation, info

    def step(self, action):
        # Take a step in the environment
        observation, reward, done, truncated, info = self.env.step(action)
        # Add noise to the observation
        noisy_observation = self.add_noise(observation)
        return noisy_observation, reward, done, truncated, info

    def add_noise(self, observation):
        # Add Gaussian noise to the observation
        noise = np.random.normal(0, self.noise_level, size=observation.shape)
        return observation + noise