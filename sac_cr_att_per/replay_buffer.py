from typing import Dict, List, Deque, Tuple
from collections import deque
import numpy as np
import torch

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

class PrioritizedReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, size: int, batch_size: int = 1024, alpha: float = 0.4, beta: float = 0.4, beta_increment: float = 0.00001, uniform_ratio = 0.3):
        self.obs_buf = np.zeros([size, n_agents, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, n_agents, obs_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, n_agents], dtype=np.float32)
        self.acts_buf = np.zeros([size, n_agents, action_dim], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.priorities = np.zeros(size, dtype=np.float32) + 1e-5  # Small value to avoid zero priorities
        self.uniform_ratio = uniform_ratio # percentage of samples that are sampled fully random
        
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance Sampling correction
        self.beta_increment = beta_increment

    def store(self, obs, act, rew, next_obs, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.priorities[self.ptr] = max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        scaled_priorities = self.priorities[:self.size] ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        per_size = int(self.batch_size * (1 - self.uniform_ratio))
        random_size = self.batch_size - per_size
        
        idx = np.random.choice(self.size, size=per_size, p=probs, replace=False)
        uniform_indices = np.random.choice(self.size, size=random_size, replace=False)

        idx = np.concatenate([idx, uniform_indices])

        weights = (self.size * probs[idx]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        self.beta = min(1.0, self.beta + self.beta_increment)  # Increase beta over time
        
        return dict(
            obs=self.obs_buf[idx],
            next_obs=self.next_obs_buf[idx],
            acts=self.acts_buf[idx],
            rews=self.rews_buf[idx],
            done=self.done_buf[idx],
            idx=idx,
            weights=torch.FloatTensor(weights).unsqueeze(1)
        )

    def update_priorities(self, idx, priorities):
        self.priorities[idx] = priorities + 1e-5  # Avoid zero priority

    def __len__(self):
        return self.size
