from bluesky_zoo import sector_cr_v0

from sac.actor import FeedForwardActor
from sac.critic_q  import FeedForward_Q
from sac.replay_buffer import ReplayBuffer
from sac.SAC import SAC

import numpy as np
import torch

def save_models(model, weights_folder = 'sac/weights'):
    torch.save(model.actor.state_dict(), weights_folder+"/actor.pt")
    torch.save(model.critic_q.state_dict(), weights_folder+"/qf.pt")
    torch.save(model.critic_q_target.state_dict(), weights_folder+"/qf_target.pt")


env = sector_cr_v0.SectorCR(render_mode=None)

action_dim = env.action_space('KL001').shape[0] 
observation_dim = env.observation_space('KL001').shape[0]
n_agents = env.num_ac 

num_episodes = 1_000

Buffer = ReplayBuffer(obs_dim = observation_dim,
                      action_dim = action_dim,
                      n_agents = n_agents,
                      size = int(1e6))

Actor = FeedForwardActor(in_dim = observation_dim,
                         out_dim = action_dim)

Critic_q = FeedForward_Q(state_dim = observation_dim,
                         action_dim = action_dim)

Critic_q_t = FeedForward_Q(state_dim = observation_dim,
                         action_dim = action_dim)

model = SAC(action_dim=action_dim,
            buffer = Buffer,
            actor = Actor,
            critic_q = Critic_q,
            critic_q_target= Critic_q_t)


observations, infos = env.reset()
agents = list(observations.keys())

obs_array = np.array(list(observations.values()))
act_array = model.get_action(obs_array)

actions = {agent: action for agent, action in zip(agents,act_array)}

observations, rewards, dones, truncates, infos = env.step(actions)

obs_array_n = np.array(list(observations.values()))
rew_array = np.array(list(rewards.values()))
done = list(dones.values())[0]

# model.store_transition(obs_array,act_array,obs_array_n,rew_array,done)

total_rew = np.array([])

for episode in range(num_episodes):
    observations, infos = env.reset()
    done = False
    rew = 0
    while not done:
        obs_array = np.array(list(observations.values()))
        act_array = model.get_action(obs_array)

        actions = {agent: action for agent, action in zip(agents,act_array)}

        observations, rewards, dones, truncates, infos = env.step(actions)

        obs_array_n = np.array(list(observations.values()))
        rew_array = np.array(list(rewards.values()))
        rew += rew_array.mean()

        if list(dones.values())[0] or list(truncates.values())[0]:
            done = True

        model.store_transition(obs_array,act_array,obs_array_n,rew_array,False)

    total_rew = np.append(total_rew,rew)
    if episode % 10 == 0:
        print(f'episode: {episode}, avg rew: {total_rew[-100:].mean()}')
        save_models(model)


import code
code.interact(local=locals())

# create the numpy arrays:
# obs_array = np.array(list(observations.values()))
# would be nice to have this in a wrapper and just create an array environment
