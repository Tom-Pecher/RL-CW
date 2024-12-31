import torch
import torch.nn as nn
from typing import List

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: List[int]):
        """
        initialize the value network to store each state's val
        """
        super().__init__()

        # Architecture for the value network 
        self.value_net = nn.Sequential(
            nn.Linear(state_dim,hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1)
        )

        # initialize weights using the hyper parameters 
        for layer in self.value_net.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.data = nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('tanh'))
                layer.bias.data.zero_()

    def forward(self, state: torch.Tensor):
        """
        forward pass of the value network 
        """
        return self.value_net(state)
