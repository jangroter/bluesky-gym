""" Example code on how to load SAC using the current folder structure"""
import numpy as np

from actor import FeedForwardActor
from critic_q  import FeedForward_Q
from replay_buffer import ReplayBuffer

from SAC import SAC

action_dim = 2 # (hdg, spd)
observation_dim = 10 # could be anything, depends on the size of your observation -> should be an array, not the dict used in gymnasium
n_agents = 1 # number of acting agents, not number of agents in the environment

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

# Random action for 1 observation, thus 1 actor
random_obs = np.random.rand(1,observation_dim)
random_action = model.get_action(random_obs)
print(random_action)

# Random action for 3 observations, thus 3 actors
random_obs = np.random.rand(3,observation_dim)
random_action = model.get_action(random_obs)
print(random_action)

""" Here a commented section with some pseudocode on what a training loop using this would look like in BS-Gym:"""
#   Initialize all the things like done above:
#
#   Actor, Critic_q, Critic_q_t, model 
#
#   for episode in training episodes:
#       new_obs, done = env.reset()
#       while not done:
#           obs = new_obs
#           action = model.get_action(obs)
#           new_obs, reward, done, truncated, info = env.step(action[()])
#           model.store_transition(obs,action,new_obs,reward,done) # this is where the training and storing of the information in the model happens
#       
#           Here you can do some per step logging for personal use
#       Here you can do some per episode logging for personal use
