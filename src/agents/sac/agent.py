"""
Agent for the SAC (Soft Actor-Critic) RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

from agents.common.replay_buffer import ReplayBuffer
from agents.common.gaussian_policy import GaussianPolicy
from agents.common.critic import Critic

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float, device: torch.device) -> None:
        self.device = device
        self.max_action = max_action

        # Training hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 100
        self.critic_lr = 3e-4
        self.actor_lr = 3e-4
        self.alpha_lr = 3e-4

        # Initialize networks
        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)

        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # Automatic entropy tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        self.memory = ReplayBuffer(1000000)

        self.max_grad_norm = 1.0  # Add gradient clipping threshold

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select an action from the policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            if evaluate:
                mean, _ = self.actor(state)
                return mean.cpu().numpy().flatten()
            else:
                action, _ = self.actor.sample(state)
                action = action.cpu().numpy().flatten()
                # Add extra exploration noise during training
                noise = np.random.normal(0, 0.1, size=action.shape)
                return np.clip(action + noise, -self.max_action, self.max_action)

    def train(self) -> float:
        """Perform one iteration of training."""
        if len(self.memory) < self.batch_size:
            return 0

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha.exp() * next_log_probs
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        critic_loss_1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_2 = F.mse_loss(current_Q2, target_Q)

        # Update first critic
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        self.critic_1_optimizer.step()

        # Update second critic
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_2_optimizer.step()

        # Update actor
        actions_new, log_probs = self.actor.sample(states)
        Q1_new = self.critic_1(states, actions_new)
        Q2_new = self.critic_2(states, actions_new)
        Q_new = torch.min(Q1_new, Q2_new)

        actor_loss = (self.log_alpha.exp() * log_probs - Q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update target networks
        self._soft_update(self.critic_1, self.critic_target_1)
        self._soft_update(self.critic_2, self.critic_target_2)

        # Update schedulers with their respective losses
        self.actor_scheduler.step(actor_loss.item())
        self.critic_1_scheduler.step(critic_loss_1.item())
        self.critic_2_scheduler.step(critic_loss_2.item())

        return critic_loss_1.item() + critic_loss_2.item()

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Perform soft update of target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
