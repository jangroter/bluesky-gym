"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import SAC
# from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.sac import MlpPolicy


from pettingzoo.sisl import waterworld_v4
from bluesky_zoo import sector_cr_v0, merge_v0


def train_sector_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Create the parallel environment
    env = env_fn(render_mode=None, **env_kwargs)
    env.reset(seed=seed)
    
    print(f"Starting training on {str(env.metadata['name'])}.")

    # Convert to vector environment
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = SAC(
        MlpPolicy,
        env,
        verbose=0,
        learning_rate=1e-3,
        batch_size=1024,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval_sector(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    env = env_fn(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = SAC.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        observations, infos = env.reset(seed=i)
        done = False
        trunc = False
        while not done and not trunc:
            actions = {agent: model.predict(obs, deterministic=True)[0] for agent, obs in zip(env.possible_agents, observations.values())}
            # import code
            # code.interact(local=locals())
            observations, rewards, dones, truncates, infos = env.step(actions)

            done = dones['KL001']
            trunc = truncates['KL001']

        
            # for a in env.agents:
            #     rewards[a] += env.rewards[a]
            # if termination or truncation:
            #     break
            # else:
            #     act = model.predict(obs, deterministic=True)[0]

            # env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":

    env_fn = merge_v0.MergeEnv
    

    # Train a model (takes ~3 minutes on GPU)
    train_sector_supersuit(env_fn, steps=10_000_000, seed=0)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval_sector(env_fn, num_games=10, render_mode='human')

    # Watch 2 games
    eval_sector(env_fn, num_games=2, render_mode="human")


# model1 = SAC
# model2 = SAC
# models = [model1, model2]

# buffer1
# ...
# ...

# for episode in training_episodes:
#     observations, infos = env.reset()
#     done = False
#     #insert code here that splits observations between model 1 - n
#     while not done:
#         action = {}
#         for model in models:
#             action_temp = {agent: model.predict(obs, deterministic=True)[0] for agent, obs in zip(model1_agents, observations.values())}

