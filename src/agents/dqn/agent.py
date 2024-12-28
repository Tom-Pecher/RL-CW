"""
The implementation of the deep Q-learning agent.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from agents.common.replay_buffer import ReplayBuffer
from agent import Agent

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent(Agent):
    def __init__(self,env_info: Dict[str,any], device):

        self.device = device
        self.action_dim =env_info['action_dim']
        state_dim = env_info['observation_dim']

        # Create neural networks
        self.policy_net = DQN(state_dim, self.action_dim).to(device)
        self.target_net = DQN(state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        self.target_update = 10
        self.batch_size = 64

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(100000)
        self.steps = 0

    @staticmethod
    def get_agent_name() -> str:
        return "dqn"

    def load_agent(self,model_path: str | Dict[str,str]) -> bool:
        if isinstance(model_path,str):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policy_net.eval()
            return True
        return False


    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.uniform(-1, 1, self.action_dim)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.cpu().data.numpy()[0]

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)

        # Get current Q values
        current_q_values = self.policy_net(states)
        # Get Q values for the actions that were actually taken
        current_q_values = torch.sum(current_q_values * actions, dim=1, keepdim=True)

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            best_actions = next_q_values.max(1)[0]
            expected_q_values = rewards + (1 - dones) * self.gamma * best_actions.unsqueeze(1)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1

        return loss.item()
