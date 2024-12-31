import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List

class PolicyNetwork(nn.Module):
    """
    a stochastic mapping for states to actions.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_sizes: List[int], upper_bound: float, lower_bound: float):
        """
        policy network to be used as a module. 
        """
        super().__init__()
        self.max_action = max_action

        # ARCHITECTURE 
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh()
        )

        # mean and log std 
        self.mean = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # initialize weights using the hyper parameters 
        for layer in self.policy.modules():
            if isinstance(layer,nn.Linear):
                layer.weight.data = nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('tanh'))
                layer.bias.data.zero_()

                
        nn.init.uniform_(self.mean.weight, -0.003, 0.003)
        nn.init.constant_(self.mean.bias, 0)
        self.log_std.data.fill_(0.0)  # begin with a zero log std.

    def forward(self, state: torch.Tensor):
        """
        works out the mean and std of the policy network 
        """
        # range for clamping, (has significant effect on convergence)
        upper_bound = 2
        lower_bound = -20
        # forward_mean = max_action * tanh(1/N sum of state policies)
        for_mean = self.max_action * torch.tanh(self.mean(self.policy(state)))  

        clamped_log_std = self.log_std.clamp(min=lower_bound,max=upper_bound)  # range should allow decent convergence quickly
        std = clamped_log_std.exp().expand_as(for_mean)
        return for_mean, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """
        just a return guassian distribution using mean, std from the forward pass. 
        """
        mean, std = self(state)
        return Normal(mean, std)