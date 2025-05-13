import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Tuple

import sac_merge.transformer as transformer
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

class MultiHeadAdditiveActorBasic(Actor):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.MultiHeadAdditiveAttentionBlockBasic(q_dim,kv_dim,num_heads)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        hidden_dims = [512,256,128]
        for hidden_dim in hidden_dims:
            layer = nn.Linear(in_dim, hidden_dim)
            nn.init.kaiming_normal_(layer.weight)
            
            self.layers.append(layer)
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.Dropout(dropout_rate))
            
            in_dim = hidden_dim



        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state):

        x, q_x, _ = self.block1(state)

        x = torch.cat((x,q_x),dim=2) # add query information also before passing through the FFN
        # Forward pass
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