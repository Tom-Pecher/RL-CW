"""
Agent for the SUNRISE (Soft UNceRtaInty-aware Safety-critical Exploration) RL.
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

class SUNRISEAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float, device: torch.device) -> None:
        self.device = device
        self.max_action = max_action
        
        # SUNRISE specific parameters
        self.num_critics = 5  # Ensemble size
        self.temperature = 20.0  # Temperature for weighted Bellman backup
        self.ucb_lambda = 1.0  # UCB exploration bonus coefficient
        
        # Standard SAC parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 100
        self.critic_lr = 3e-4
        self.actor_lr = 3e-4
        self.alpha_lr = 3e-4

        # Initialize actor
        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
        
        # Initialize ensemble of critics and their targets
        self.critics = []
        self.critic_targets = []
        self.critic_optimizers = []
        
        for _ in range(self.num_critics):
            critic = Critic(state_dim, action_dim).to(device)
            self.critics.append(critic)
            self.critic_targets.append(copy.deepcopy(critic))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=self.critic_lr))

        # Initialize actor optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Automatic entropy tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # Initialize replay buffer
        self.memory = ReplayBuffer(1000000)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select an action using UCB exploration."""
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            if evaluate:
                mean, _ = self.actor(state)
                return mean.cpu().numpy().flatten()
            
            # Sample multiple actions
            num_samples = 10
            actions = []
            for _ in range(num_samples):
                action, _ = self.actor.sample(state)
                actions.append(action)
            actions = torch.cat(actions, dim=0)
            
            # Get Q-values from all critics
            q_values = []
            for critic in self.critics:
                q = critic(state.repeat(num_samples, 1), actions)
                q_values.append(q)
            q_values = torch.stack(q_values, dim=0)
            
            # Calculate mean and std of Q-values
            mean_q = q_values.mean(dim=0)
            std_q = q_values.std(dim=0)

            # UCB action selection
            ucb_values = mean_q + self.ucb_lambda * std_q
            best_action_idx = ucb_values.argmax()

            return actions[best_action_idx].cpu().numpy().flatten()

    def train(self) -> float:
        """Perform one iteration of training."""
        if len(self.memory) < self.batch_size:
            return 0

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Update critics
        total_critic_loss = 0
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Get target Q-values from all critics
            target_qs = []
            for critic_target in self.critic_targets:
                target_q = critic_target(next_states, next_actions)
                target_qs.append(target_q)
            target_qs = torch.stack(target_qs, dim=0)

            # Weighted Bellman backup
            weights = F.softmax(-target_qs / self.temperature, dim=0)
            target_q = (weights * target_qs).sum(dim=0) - self.log_alpha.exp() * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Update each critic
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            current_q = critic(states, actions)
            critic_loss = F.mse_loss(current_q, target_q)

            optimizer.zero_grad()
            critic_loss.backward()
            optimizer.step()
            
            total_critic_loss += critic_loss.item()

        # Update actor
        actions_new, log_probs = self.actor.sample(states)
        q_values = []
        for critic in self.critics:
            q = critic(states, actions_new)
            q_values.append(q)
        q_values = torch.stack(q_values, dim=0)
        q_mean = q_values.mean(dim=0)
        
        actor_loss = (self.log_alpha.exp() * log_probs - q_mean).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update target networks
        for critic, critic_target in zip(self.critics, self.critic_targets):
            self._soft_update(critic, critic_target)

        return total_critic_loss / self.num_critics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Perform soft update of target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )