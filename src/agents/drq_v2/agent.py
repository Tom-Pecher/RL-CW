"""
DrQ-v2 agent implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from agents.common.utils import schedule
from agents.common.encoder import Encoder
from agents.sac.agent import SACAgent


class DrQv2Agent(SACAgent):
    def __init__(self, state_dim, action_dim, max_action, device):
        super().__init__(state_dim, action_dim, max_action, device)

        # Override SAC's tau
        self.tau = self.critic_target_tau = 0.01

        # Training hyperparameters (mostly same as SAC)
        self.update_every_steps = 2
        self.num_expl_steps = 2000
        self.stddev_schedule = 'linear(1.0,0.1,50000)'
        self.stddev_clip = 0.3
        self.noise_level = 0.1
        
        # Initialize shared encoder
        self.encoder = Encoder(state_dim).to(device)
        self.encoder_target = Encoder(state_dim).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        
        # Use specific architecture for actor and critic
        self.actor = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim * 2)
        ).to(device)
        
        self.critic = nn.Sequential(
            nn.Linear(1024 + action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        ).to(device)
        
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.critic_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
    def augment(self, x):
        """Apply random shifts for data augmentation."""
        noise = torch.randn_like(x) * self.noise_level
        return x + noise
    
    def select_action(self, state, evaluate=False):
        """Select action for given state."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            h = self.encoder(state)
            
            if evaluate or self.total_steps > self.num_expl_steps:
                action, _ = self.actor.sample(h)
            else:
                # Use scheduled exploration in early steps
                stddev = schedule(self.stddev_schedule, self.total_steps)
                dist = self.actor.get_dist(h)
                action = dist.mean + torch.randn_like(dist.mean) * stddev
                action = torch.clamp(action, -self.stddev_clip, self.stddev_clip)
            
            return action.cpu().data.numpy().flatten()
    
    def train(self):
        # Skip updates based on update_every_steps
        if self.total_steps % self.update_every_steps != 0:
            return 0
            
        if len(self.memory) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )
        
        # Augment states and next_states
        states = self.augment(states)
        next_states = self.augment(next_states)
        
        # Encode states
        h = self.encoder(states)
        with torch.no_grad():
            h_target = self.encoder_target(next_states)
            
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(h_target)
            target_Q = self.critic_target(torch.cat([h_target, next_actions], dim=1))
            target_Q = rewards + (1 - dones) * self.gamma * (
                target_Q - self.log_alpha.exp() * next_log_probs
            )
            
        current_Q = self.critic(torch.cat([h, actions], dim=1))
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Update encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.encoder_opt.step()
        self.critic_opt.step()
        
        # Update actor
        actions_new, log_probs = self.actor.sample(h.detach())
        Q_new = self.critic(torch.cat([h.detach(), actions_new], dim=1))
        actor_loss = (self.log_alpha.exp() * log_probs - Q_new).mean()
        
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha.exp() * 
                      (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        self._soft_update(self.encoder, self.encoder_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item()
