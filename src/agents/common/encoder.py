"""
Shared encoder
"""


import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.trunk(x)