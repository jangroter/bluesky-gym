from bluesky_zoo import sector_cr_v0

from sac.actor import FeedForwardActor
from sac.critic_q  import FeedForward_Q
from sac.replay_buffer import ReplayBuffer
from sac.SAC import SAC

import numpy as np

env = sector_cr_v0.SectorCR(render_mode=None)

action_dim = env.action_space('KL001').shape[0] 
observation_dim = env.observation_space('KL001').shape[0]
n_agents = env.num_ac 

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

model.store_transition(obs_array,act_array,obs_array_n,rew_array,done)

import code
code.interact(local=locals())

# create the numpy arrays:
# obs_array = np.array(list(observations.values()))
# would be nice to have this in a wrapper and just create an array environment
