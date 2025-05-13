import torch
import torch.nn.functional as F
import torch.optim as optim

from typing import Tuple, Optional, List
import numpy as np

from sac_merge.actor import Actor
from sac_merge.critic_q import Critic_Q
from sac_merge.replay_buffer import ReplayBuffer


class SAC:
    def __init__(self, 
                 action_dim, 
                 buffer, 
                 actor, 
                 critic_q, 
                 critic_q_target, 
                 alpha_lr=3e-4, 
                 actor_lr=3e-4, 
                 critic_q_lr=3e-3,
                 gamma=0.995, 
                 tau=5e-3, 
                 policy_update_freq=10, 
                 initial_random_steps=0):
        self.gamma = gamma
        self.tau = tau
        self.test = False
        self.initial_random_steps = initial_random_steps
        self.total_steps = 0
        self.device = torch.device("cpu")
        self.action_dim = action_dim
        self.buffer = buffer

        self.target_alpha = -np.prod((self.action_dim)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.qf1_lossarr = np.array([])
        self.qf2_lossarr = np.array([])

        self.policy_update_freq = policy_update_freq
        self.actor = actor.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic_q = critic_q.to(self.device)
        self.critic_optim = optim.Adam(self.critic_q.parameters(), lr=critic_q_lr)
        self.critic_q_target = critic_q_target.to(self.device)
        
        self.hard_update(self.critic_q_target, self.critic_q)
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        if self.total_steps < self.initial_random_steps and not self.test:
            action = np.random.standard_normal((len(observation),self.action_dim)) * 0.33
        else:
            action = self.actor(torch.FloatTensor(np.array([observation])).to(self.device))[0].detach().cpu().numpy()
            action = np.array(action[0])
            action = np.clip(action, -1, 1)
        
        self.total_steps += 1
        return action

    def store_transition(self,observation,action,new_observation,reward,done) -> None:
        if not self.test:
            done = False
            transition = [observation, action, reward, new_observation, done]
            self.buffer.store(*transition)
        if (self.total_steps % self.policy_update_freq == 0 and 
            len(self.buffer) >  self.buffer.batch_size and 
            self.total_steps > self.initial_random_steps and 
            not self.test):
                self.update_model()


    def new_episode(self, test: bool) -> None:
        self.test = test

    def update_model(self):
        device = self.device
        samples = self.buffer.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"]).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        weights = samples["weights"].to(device)
        idx = samples["idx"]

        b,n = reward.size()
        reward = reward.view(b,n,1)

        alpha = self.log_alpha.exp()
        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor(next_state)
            qf_target = self.critic_q_target(next_state, next_state_action)
            qf1_next_target = qf_target[:,:,0]
            qf2_next_target = qf_target[:,:,1]
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi.flatten(start_dim=-2,end_dim=-1)
            next_q_value = reward.flatten(start_dim=-2,end_dim=-1) + self.gamma * (min_qf_next_target)
        
        qf = self.critic_q(state, action)
        qf1_loss = F.mse_loss(qf[:,:,0], next_q_value, reduction='none')
        qf2_loss = F.mse_loss(qf[:,:,1], next_q_value, reduction='none')
        qf_loss = (qf1_loss + qf2_loss) * weights

        self.critic_optim.zero_grad()
        qf_loss.mean().backward()
        self.critic_optim.step()

        td_error = (torch.abs(qf[:,:,0] - next_q_value).detach() + torch.abs(qf[:,:,1] - next_q_value).detach()).sum(dim=1) + 1e-5
        self.buffer.update_priorities(idx, td_error.cpu().numpy())

        pi, log_pi = self.actor(state)
        min_qf_pi = torch.min(self.critic_q(state, pi)[:,:,0], self.critic_q(state, pi)[:,:,1])
        policy_loss = ((alpha * log_pi.flatten(-2,-1)) - min_qf_pi).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.critic_q_target, self.critic_q, self.tau)

        alpha_loss = -(self.log_alpha * (log_pi + self.target_alpha).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return policy_loss.item(), alpha_loss.item()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)