import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Tuple

def query_key_from_state(state):
    # Extract req. info from observation
    positions = torch.tensor(state[:,:,8:10])
    velocities = torch.tensor(state[:,:,10:12])
    speed = torch.norm(velocities, dim=-1, keepdim=True) 
    direction = velocities / (speed + 1e-8)

    b, t, _ = positions.size()

    # Determine relative pos & vel from perspective of agent
    relative_positions = positions.unsqueeze(1) - positions.unsqueeze(2)
    relative_velocities = velocities.unsqueeze(1) - velocities.unsqueeze(2)

    # Create rotation matrix
    rot_x = direction
    rot_y = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    rotation_matrix = torch.stack([rot_x, rot_y], dim=-2)

    # Apply the rotation
    rel_pos_rotated = torch.matmul(rotation_matrix.unsqueeze(2), relative_positions.unsqueeze(-1)).squeeze(-1)
    rel_vel_rotated = torch.matmul(rotation_matrix.unsqueeze(2), relative_velocities.unsqueeze(-1)).squeeze(-1)

    # Removing Self-References
    mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
    rel_pos_rotated = rel_pos_rotated[mask].reshape(b, t, t-1, 2)
    rel_vel_rotated = rel_vel_rotated[mask].reshape(b, t, t-1, 2)
    relative_states = torch.cat([rel_pos_rotated, rel_vel_rotated], dim=-1)

    # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
    r = torch.sqrt(relative_states[:,:,:,0].clone()**2 + relative_states[:,:,:,1].clone()**2).view(b,t,t-1,1)

    # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
    r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))
    
    # Add transformed distance vector to the state vector
    relative_states = torch.cat((relative_states,r_trans),dim=-1)

    # Add track_error of the agents to the keys for intent information
    track_error = torch.tensor(state[:,:,0:2])
    track_error_expanded = track_error.unsqueeze(1).expand(-1, t, -1, -1)
    track_error_selected = track_error_expanded[mask].reshape(b, t, t-1, 2)
    relative_states = torch.cat([relative_states, track_error_selected], dim=-1)

    q_x = state[:,:,0:8]
    kv_x = relative_states

    return q_x, kv_x

### ATTENTION MODULES

class RelativeAdditiveAttentionMultiHead(nn.Module):
    """
    for how multihead additive
    **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """

    def __init__(self, q_dim, kv_dim, num_heads=3):
        super(RelativeAdditiveAttentionMultiHead, self).__init__()

        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        # Generate the transformer matrices
        self.num_heads = num_heads

        self.tokeys    = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)
        self.toqueries = nn.Linear(q_dim, kv_dim*num_heads, bias=False)
        self.tovalues  = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)

        self.bias = nn.Parameter(torch.rand(kv_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(kv_dim, 1)
    
    def forward(self, q_x, kv_x):
        b, t,_, k = kv_x.size()
        h = self.num_heads
        
        keys = self.tokeys(kv_x).view(b,t,(t-1),h,k)
        queries =  self.toqueries(q_x).view(b,t,h,k)
        queries = queries.view(b,t,1,h,k)

        score = self.score_proj(torch.tanh(keys + queries + self.bias)).squeeze(-1)
        
        values = self.tovalues(kv_x)
        values = values.view(b,t,t-1,h,k)
        
        w = F.softmax(score, dim=2)

        w = w.transpose(2,3).view(b,t,h,1,t-1)
        v = values.transpose(2,3)

        x = torch.matmul(w,v)
        x = x.view(b,t,k*h)
        return x

## ADDITIVE

class MultiHeadAdditiveAttentionBlockQ(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadAdditiveAttentionBlockQ, self).__init__()

        self.att = RelativeAdditiveAttentionMultiHead(q_dim, kv_dim, num_heads)

        self.norm1 = nn.LayerNorm(kv_dim * num_heads)

        self.ff = nn.Sequential(
            nn.Linear(kv_dim * num_heads, 5 * kv_dim * num_heads),
            nn.ReLU(),
            nn.Linear(5 * kv_dim * num_heads, kv_dim * num_heads))
        
        self.norm2 = nn.LayerNorm(kv_dim * num_heads)

    def forward(self, state, action):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = query_key_from_state(state)
        q_x = torch.cat((q_x, action), dim=-1)

        # Self-attend
        y = self.att(q_x,kv_x)

        # First residual connection
        # x = q_x + y
        x = y

        # Normalize
        x = self.norm1(x)

        # Pass through feed-forward network
        y = self.ff(x)

        # Second residual connection
        x = x + y

        # Again normalize
        y = self.norm2(x)

        return y, q_x, kv_x
    
    def get_input_tensors(self, state):
        rel_state_init = state[:,:,0:4]

        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)
        
        # Create x,y tensors for each aircraft by doubling the k dimension
        xy_state = rel_state.view(b,t,t,2,2).transpose(3,4)
        angle = state[:,:,-1]
        angle = -angle + 0.5 * torch.pi
        angle = angle.view(b,t,1)
        x_new = xy_state[:,:,:,0,:] * torch.cos(angle.view(b,1,t,1)) - xy_state[:,:,:,1,:]*torch.sin(angle.view(b,1,t,1))
        y_new = xy_state[:,:,:,0,:] * torch.sin(angle.view(b,1,t,1)) + xy_state[:,:,:,1,:]*torch.cos(angle.view(b,1,t,1))
        rel_state = torch.cat((x_new,y_new),dim=3).view(b,t,t,2,2).transpose(3,4).reshape(b,t,t,4)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
        r = torch.sqrt(rel_state[:,:,:,0].clone()**2 + rel_state[:,:,:,1].clone()**2).view(b,t,t-1,1)
        
        # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
        r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))

        # divide x and y by the distance to get the values as a function of the unit circle
        rel_state[:,:,:,0:2] = rel_state[:,:,:,0:2].clone()/r
        
        # add transformed distance vector to the state vector
        rel_state = torch.cat((rel_state,r_trans),dim=-1)

        q_x = state
        kv_x = rel_state

        return q_x, kv_x

class MultiHeadAdditiveAttentionBlockQBasic(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadAdditiveAttentionBlockQBasic, self).__init__()

        self.att = RelativeAdditiveAttentionMultiHead(q_dim, kv_dim, num_heads)

    def forward(self, state, action):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = query_key_from_state(state)
        q_x = torch.cat((q_x, action), dim=-1)

        # Self-attend
        y = self.att(q_x,kv_x)

        return y, q_x, kv_x

class MultiHeadAdditiveAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadAdditiveAttentionBlock, self).__init__()

        self.att = RelativeAdditiveAttentionMultiHead(q_dim, kv_dim, num_heads)

        self.norm1 = nn.LayerNorm(kv_dim * num_heads)

        self.ff = nn.Sequential(
            nn.Linear(kv_dim * num_heads, 5 * kv_dim * num_heads),
            nn.ReLU(),
            nn.Linear(5 * kv_dim * num_heads, kv_dim * num_heads))
        
        self.norm2 = nn.LayerNorm(kv_dim * num_heads)

    def forward(self, state):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = query_key_from_state(state)(state)

        # Self-attend
        y = self.att(q_x,kv_x)

        # First residual connection
        # x = q_x + y
        x = y

        # Normalize
        x = self.norm1(x)

        # Pass through feed-forward network
        y = self.ff(x)

        # Second residual connection
        x = x + y

        # Again normalize
        y = self.norm2(x)

        return y, q_x, kv_x
    
    def get_input_tensors(self, state):
        rel_state_init = state[:,:,0:4]

        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)
        
        # Create x,y tensors for each aircraft by doubling the k dimension
        xy_state = rel_state.view(b,t,t,2,2).transpose(3,4)
        angle = state[:,:,-1]
        angle = -angle + 0.5 * torch.pi
        angle = angle.view(b,t,1)
        x_new = xy_state[:,:,:,0,:] * torch.cos(angle.view(b,1,t,1)) - xy_state[:,:,:,1,:]*torch.sin(angle.view(b,1,t,1))
        y_new = xy_state[:,:,:,0,:] * torch.sin(angle.view(b,1,t,1)) + xy_state[:,:,:,1,:]*torch.cos(angle.view(b,1,t,1))
        rel_state = torch.cat((x_new,y_new),dim=3).view(b,t,t,2,2).transpose(3,4).reshape(b,t,t,4)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
        r = torch.sqrt(rel_state[:,:,:,0].clone()**2 + rel_state[:,:,:,1].clone()**2).view(b,t,t-1,1)
        
        # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
        r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))

        # divide x and y by the distance to get the values as a function of the unit circle
        rel_state[:,:,:,0:2] = rel_state[:,:,:,0:2].clone()/r
        
        # add transformed distance vector to the state vector
        rel_state = torch.cat((rel_state,r_trans),dim=-1)

        q_x = state
        kv_x = rel_state

        return q_x, kv_x

class MultiHeadAdditiveAttentionBlockBasic(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadAdditiveAttentionBlockBasic, self).__init__()

        self.att = RelativeAdditiveAttentionMultiHead(q_dim, kv_dim, num_heads)

    def forward(self, state):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = query_key_from_state(state)
        y = self.att(q_x,kv_x)

        return y, q_x, kv_x
