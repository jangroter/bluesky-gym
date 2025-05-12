from typing import Dict, List, Deque, Tuple
from collections import deque
import numpy as np

class ReplayBuffer:

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, size: int, batch_size: int = 1024):
        self.obs_buf = np.zeros([size, n_agents, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, n_agents, obs_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, n_agents], dtype=np.float32)
        self.acts_buf = np.zeros([size, n_agents, action_dim], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """ Store transition """
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """ Sample from storage"""
        idx = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs = self.obs_buf[idx],
            next_obs = self.next_obs_buf[idx],
            acts = self.acts_buf[idx],
            rews = self.rews_buf[idx],
            done = self.done_buf[idx])
    
    def __len__(self) -> int:
        return self.size