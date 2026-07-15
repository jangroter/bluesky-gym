"""
Minimal learning check for the multi-agent competition environment.

Trains a single shared policy across all agents (parameter sharing) with PPO,
and prints the mean per-agent episode return before vs. after training. This is
a sanity check that the reward signal is learnable, not a tuned training setup.

Parameter sharing is done with SuperSuit's ``pettingzoo_env_to_vec_env_v1``,
which exposes the N homogeneous agents as an N-way vectorized env backed by a
*single* underlying ParallelEnv instance. That single-instance property is what
lets this work with BlueSky, which is a process-global singleton.

Run:
    .venv/bin/python train_zoo.py [total_timesteps]     # train, then save to MODEL_PATH
    .venv/bin/python train_zoo.py replay [model_path]   # load MODEL_PATH, render (human)

"""

import sys

import numpy as np
import supersuit as ss
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from bluesky_zoo.competition_v0 import CompetitionZooEnv


class FlattenObs(BaseParallelWrapper):
    """Flatten each agent's Dict observation into a 1-D Box.

    SuperSuit's flatten_v0 only flattens multi-dimensional Box spaces, not
    Dict spaces, so we do the Dict->vector flattening here (via gymnasium's
    space flattening) before handing the env to the SuperSuit vector stack.
    """

    def __init__(self, env):
        super().__init__(env)
        self._flat = {a: spaces.flatten_space(env.observation_space(a))
                      for a in env.possible_agents}

    def observation_space(self, agent):
        return self._flat[agent]

    def _flatten(self, observations):
        return {a: spaces.flatten(self.env.observation_space(a), obs)
                for a, obs in observations.items()}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._flatten(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._flatten(observations), rewards, terminations, truncations, infos


SEED = 0
N_AGENTS = 3
EVAL_EPISODES = 12   # per-agent episodes to average over
MODEL_PATH = "ppo_competition"


def make_vec_env():
    env = CompetitionZooEnv(render_mode=None, n_agents=N_AGENTS)
    env = FlattenObs(env)          # Dict obs -> flat Box, so plain MlpPolicy works
    env = ss.black_death_v3(env)   # keep terminated agents (zero-padded) until episode end
    env = ss.pettingzoo_env_to_vec_env_v1(env)   # -> N sub-envs, one shared policy
    env = SB3VecEnvWrapper(env)    # SB3 VecEnv API directly (NOT concat_vec_envs_v1; see module docstring)
    return VecMonitor(env)         # records per-episode returns in info["episode"]


def evaluate(venv, n_episodes, model=None):
    """ Average per-agent episode return; random actions when model is None.
    """
    returns = []
    obs = venv.reset()
    while len(returns) < n_episodes:
        if model is None:
            actions = np.stack([venv.action_space.sample() for _ in range(venv.num_envs)])
        else:
            actions, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = venv.step(actions)
        for info in infos:
            if "episode" in info:            # VecMonitor logs a completed sub-env episode
                returns.append(info["episode"]["r"])
    return float(np.mean(returns)), float(np.std(returns))


def main(total_timesteps):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    venv = make_vec_env()

    # Reference value before training to compare against
    base_mean, base_std = evaluate(venv, EVAL_EPISODES)
    print(f"\nBEFORE (random policy):  mean episode return = {base_mean:8.2f} +/- {base_std:.2f}")

    # Define and train model
    model = PPO(
        "MlpPolicy", venv,
        n_steps=512, batch_size=128, gae_lambda=0.95, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(MODEL_PATH)

    # After training evaluation and comparisson against random policy
    trained_mean, trained_std = evaluate(venv, EVAL_EPISODES, model=model)
    print(f"\nBEFORE (random policy):        mean episode return = {base_mean:8.2f} +/- {base_std:.2f}")
    print(f"AFTER  ({total_timesteps} steps, greedy):  mean episode return = {trained_mean:8.2f} +/- {trained_std:.2f}")
    print("learned something" if trained_mean > base_mean else "no improvement")
    print(f"saved policy to {MODEL_PATH}.zip  (replay: python train_zoo.py replay)")

    venv.close()


def replay(model_path=MODEL_PATH, n_episodes=3, seed=SEED):
    """Load the saved shared policy and render its greedy behaviour (human mode).
    """

    model = PPO.load(model_path)
    env = FlattenObs(CompetitionZooEnv(render_mode="human", n_agents=N_AGENTS))
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        while env.agents:
            agents = list(env.agents)
            batch = np.stack([obs[a] for a in agents])
            act, _ = model.predict(batch, deterministic=True)
            actions = {a: act[i] for i, a in enumerate(agents)}
            obs, _, _, _, infos = env.step(actions)
    env.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "replay":
        model_path = args[1] if len(args) > 1 else MODEL_PATH
        print(f"replaying trained model saved under {model_path}")
        replay(model_path)
    else:
        main(int(args[0]) if args else 250_000)
