
import torch.nn as nn
from torch.distributions import Normal
import torch
class SharedNetwork(nn.Module):
    def __init__(self, state_dim):
        super(SharedNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
        )
        for layer in self.network.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.data = nn.init.orthogonal_(layer.weight.data,gain=nn.init.calculate_gain('tanh'))
                layer.bias.data.zero_()

    def forward(self, x):
        return self.network(x)


class ValueNetwork(nn.Module):
    def __init__(self,shared_net):
        super(ValueNetwork, self).__init__()
        self.shared_net = shared_net
        self.value_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.value_head.bias)


    def forward(self, x):
        shared_features = self.shared_net.forward(x)
        return self.value_head(shared_features)

class PolicyNetwork(nn.Module):
    def __init__(self,shared_net,action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.shared_net = shared_net
        self.max_action = max_action

        self.mean = nn.Linear(64,action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) 

        nn.init.uniform_(self.mean.weight, -0.003, 0.003)
        nn.init.constant_(self.mean.bias, 0)
        self.log_std.data.fill_(0.0)  # begin with a zero log std.
        
    def forward(self, x):
        upper_bound = 2
        lower_bound = -20
        shared_features = self.shared_net.forward(x) 

        mean = self.max_action * torch.tanh(self.mean(shared_features))
        clamped_log_std = self.log_std.clamp(min=lower_bound,max=upper_bound)
        std = clamped_log_std.exp().expand_as(mean)
        return mean, std 
    def get_distribution(self, state: torch.Tensor) -> Normal:
        """
        just a return guassian distribution using mean, std from the forward pass. 
        """
        mean, std = self(state)
        return Normal(mean, std)
    

