import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from abc import ABC, abstractmethod

import sac_cr_att.transformer as transformer

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Critic_Q(ABC, nn.Module):

    def __init__(self):
        super(Critic_Q, self).__init__()
    
    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

class MultiHeadAdditiveCriticQv3Basic(Critic_Q):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.MultiHeadAdditiveAttentionBlockQBasic(q_dim,kv_dim,num_heads)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256

        self.out = nn.Linear(in_dim, 2)
        self.out = init_layer_uniform(self.out)

    def forward(self, state, action):

        x, q_x, _ = self.block1(state,action)
        x = torch.cat((x,q_x),dim=-1) # add query and action information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        value = self.out(x)
        
        return value