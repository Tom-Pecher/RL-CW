"""
Agent for PPO (Proximal Policy Optimization) RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

from agents.common.gaussian_policy import GaussianPolicy


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float, device: torch.device) -> None:
        """ Initialize the PPO agent. """
        self.device = device
        self.max_action = max_action

        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5

        # Increase batch size and epochs for better stability
        self.batch_size = 256  # Increased from 64
        self.num_epochs = 10
        self.trajectory_size = 2048

        # Adjust learning rates
        self.policy_lr = 3e-4
        self.value_lr = 3e-4  # Same as policy_lr for better balance

        # Policy network (Actor)
        self.policy = GaussianPolicy(state_dim, action_dim, max_action).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

        # Value network (Critic)
        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.value_lr)

        # Storage for trajectory data
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Select an action using the policy network.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log probability, value)
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            action, log_prob = self.policy.sample(state)
            value = self.value(state)

            return (
                action.cpu().numpy()[0],
                log_prob.cpu().numpy()[0],
                value.cpu().numpy()[0]
            )

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store a transition in the trajectory buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def _compute_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).

        Returns:
            Tuple of (advantages, returns)
        """
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        # Compute last value if not done
        with torch.no_grad():
            last_value = 0.0 if self.dones[-1] else self.value(states[-1]).item()

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_gae = 0
        last_return = last_value

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            last_return = rewards[t] + self.gamma * next_non_terminal * last_return

            advantages[t] = last_gae
            returns[t] = last_return

        return advantages, returns

    def train(self) -> Tuple[float, float, float]:
        """
        Train the agent using the collected trajectory.

        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        if len(self.states) < self.trajectory_size:
            return 0.0, 0.0, 0.0

        # Convert trajectory data to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)

        # Compute advantages and returns
        advantages, returns = self._compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.num_epochs):
            # Generate random permutation for minibatches
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get minibatch
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                advantage_batch = advantages[batch_indices]
                return_batch = returns[batch_indices]

                # Get current policy distribution and value prediction
                mean, log_std = self.policy(state_batch)
                dist = torch.distributions.Normal(mean, log_std.exp())
                new_log_prob = dist.log_prob(action_batch).sum(dim=1, keepdim=True)

                # Calculate probability ratio and clip it
                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                ratio = ratio.clamp(0.0, 10.0)  # Avoid extreme ratios

                # Policy loss with clipping
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping (like in PPO paper)
                value_pred = self.value(state_batch)
                value_pred_clipped = self.values[batch_indices] + (value_pred - self.values[batch_indices]).clamp(
                    1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                value_loss1 = (value_pred - return_batch) ** 2
                value_loss2 = (value_pred_clipped - return_batch) ** 2
                value_loss = self.value_loss_coef * torch.max(value_loss1, value_loss2).mean()

                # Entropy bonus for exploration
                entropy = dist.entropy().mean() * self.entropy_coef

                # Total loss
                loss = policy_loss + value_loss - entropy

                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)

                self.policy_optimizer.step()
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # Clear trajectory buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        num_updates = self.num_epochs * (len(states) // self.batch_size)
        return (
            total_policy_loss / num_updates,
            total_value_loss / num_updates,
            total_entropy / num_updates
        ) 
