"""

"""

import functools
from pettingzoo import ParallelEnv
import gymnasium as gym
import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

import bluesky_gym.envs.common.functions as fn

class PathPlanning(ParallelEnv):
    
    metadata = {
        "name": "sector_cr_v0",
        "render_modes": ["rgb_array","human"], 
        "render_fps": 120
    }

    def __init__(self, render_mode=None, n_agents=10):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def _get_observation(self):
        pass

    def _get_reward(self):
        pass

    def _do_action(self, actions):
        pass

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent] # have to define observation_spaces & action_spaces, probably in init

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def _get_agents(self, n_agents):
        return [f'kl00{i+1}'.upper() for i in range(n_agents)]
    
    def _generate_ac(self) -> None:
        pass

    def _check_intrusion(self, ac_idx):
        pass

    def _get_dones(self):
        pass

    def _get_info(self):
        pass

    def _render_frame(self):
        pass
