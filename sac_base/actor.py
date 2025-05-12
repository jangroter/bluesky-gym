import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Tuple

from abc import ABC, abstractmethod

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Actor(ABC, nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.test = False
    
    @abstractmethod
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def set_test(self, test: bool) -> None:
        self.test = test

class FeedForwardActor(Actor):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layers = nn.ModuleList()

        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        log_std_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(hidden_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = observation
        for layer in self.layers:
            x = layer(x)

        mu =  self.mu_layer(x)#.tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu.tanh(), log_prob
        
        return action, log_prob