"""
Agent for the A2C (Advantage Actor Critic) RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.common.actor import Actor
from agents.common.critic import Critic

class A2CAgent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        self.max_action = max_action
        self.action_dim = action_dim

        # Training hyperparameters
        self.gamma = 0.99
        self.lr = 3e-4
        self.entropy_coef = 0.01

        # Actor and Critic models
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Logging attributes
        self.training = False
        self.total_steps = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
            # Add exploration noise during training
            if self.training:
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -self.max_action, self.max_action)
            return action.cpu().numpy().flatten()

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self, trajectories):
        states, actions, rewards, dones, _ = trajectories
        self.training = True

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        returns = self.compute_returns(rewards, dones)

        # Critic update
        predicted_actions = self.actor(states).detach()  # Detach actions for critic update
        state_values = self.critic(states, predicted_actions).squeeze()
        critic_loss = nn.MSELoss()(state_values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        predicted_actions = self.actor(states)
        state_values = self.critic(states, predicted_actions).squeeze()
        advantages = returns - state_values.detach()

        # Compute policy loss
        action_mean = predicted_actions
        action_std = torch.ones_like(action_mean).to(self.device) * 0.1
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Logging
        self.total_steps += len(states)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }
