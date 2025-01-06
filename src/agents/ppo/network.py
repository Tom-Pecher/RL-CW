 

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class SharedNetwork(nn.Module):
    def __init__(self, state_dim: int):
        """
        Shared network that extracts features from the state.
        """
        super(SharedNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # Initialize weights and biases for the linear layers
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the shared network.
        """
        return self.network(x)


class ValueNetwork(nn.Module):
    def __init__(self, shared_net: SharedNetwork):
        """
        Value network to predict the state value.
        """
        super(ValueNetwork, self).__init__()
        self.shared_net = shared_net
        self.value_head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get the state value.
        """
        shared_features = self.shared_net(x)
        return self.value_head(shared_features)


class PolicyNetwork(nn.Module):
    def __init__(self, shared_net: SharedNetwork, action_dim: int, max_action: float):
        """
        Policy network to output action distribution.
        """
        super(PolicyNetwork, self).__init__()
        self.shared_net = shared_net
        self.max_action = max_action

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.full((1, action_dim), 0.0))  # Start with a small log_std

        nn.init.uniform_(self.mean.weight, -0.003, 0.003)
        nn.init.zeros_(self.mean.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute action mean and standard deviation.
        """
        
        shared_features = self.shared_net(x)

        mean = self.max_action * torch.tanh(self.mean(shared_features))
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, state: torch.Tensor) -> MultivariateNormal:
        """
        Returns a Gaussian distribution using the mean and std from the forward pass.
        """
        mean, std = self(state)
        return MultivariateNormal(mean, std)

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class SharedNetwork(nn.Module):
    def __init__(self, state_dim: int):
        """
        Shared network that extracts features from the state.
        """
        super(SharedNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # Initialize weights and biases for the linear layers
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the shared network.
        """
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self, shared_net: SharedNetwork):
        """
        Value network to predict the state value.
        """
        super(ValueNetwork, self).__init__()
        self.shared_net = shared_net
        self.value_head = nn.Linear(256, 1)
        
        # Initialize the value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)  # Gain for linear layers is 1.0
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get the state value.
        """
        shared_features = self.shared_net(x)
        return self.value_head(shared_features)

class PolicyNetwork(nn.Module):
    def __init__(self, shared_net: SharedNetwork, action_dim: int, max_action: float):
        """
        Policy network to output action distribution.
        """
        super(PolicyNetwork, self).__init__()
        self.shared_net = shared_net
        self.max_action = max_action

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Per-action log_std

        # Initialize the mean layer
        nn.init.uniform_(self.mean.weight, -0.003, 0.003)
        nn.init.zeros_(self.mean.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute action mean and standard deviation.
        """
        shared_features = self.shared_net(x)
        mean = self.max_action * torch.tanh(self.mean(shared_features))
        std = self.log_std.exp()  # Ensure std is positive
        return mean, std

    def get_distribution(self, state: torch.Tensor) -> MultivariateNormal:
        """
        Returns a Gaussian distribution using the mean and std from the forward pass.
        """
        mean, std = self(state)
        cov_matrix = torch.diag_embed(std.pow(2))  # Create diagonal covariance matrix
        return MultivariateNormal(mean, cov_matrix)

# Example instantiation and usage
if __name__ == "__main__":
    state_dim = 10
    action_dim = 2
    max_action = 1.0

    shared_net = SharedNetwork(state_dim)
    value_net = ValueNetwork(shared_net)
    policy_net = PolicyNetwork(shared_net, action_dim, max_action)

    # Dummy input
    state = torch.randn(3, state_dim)  # Batch of 3 states

    # Forward passes
    value = value_net(state)
    print("State Value:", value)

    mean, std = policy_net(state)
    print("Action Mean:", mean)
    print("Action Std:", std)

    dist = policy_net.get_distribution(state)
    print("Sampled Actions:", dist.sample())
