"""
Agent for the TD3 (Twin Delayed Deep Deterministic Policy Gradient) RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agent import Agent
from agents.common.actor import Actor
from agents.common.critic import Critic
from agents.common.replay_buffer import ReplayBuffer


class TD3Agent(Agent):
    def __init__(self,env_info: Dict[str,any], device: torch.device) -> None:
        """ Initialize the TD3 agent. """

        self.device = device

        state_dim = env_info['observation_dim']
        action_dim = env_info['action_dim']
        self.max_action = float(env_info['action_high'][0])
        # Training hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.exploration_noise = 0.1
        self.batch_size = 256
        self.policy_delay = 2

        self.actor_lr = 3e-4
        self.critic_lr = 3e-4

        self.training = False
        self.total_it = 0

        # Actor
        self.actor = Actor(state_dim, action_dim, self.max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Critics
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        self.memory = ReplayBuffer(1000000)
 
    @staticmethod
    def get_agent_name() -> str:
        return "td3"
 
    def load_agent(self, model_path: str | Dict[str, str]) -> bool:
        
        if isinstance(model_path,str):
            # It might be worth try catching it 
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.actor.eval()
            return True 

        if not model_path: # for some reason this checks if a dict is empty  
            return False

        for path, path_type in model_path: 
            match path_type:
                case "actor":
                    self.actor.load_state_dict(torch.load(path, map_location=self.device))
                    self.actor.eval()
                case "critic_1":
                    self.critic_1.load_state_dict(torch.load(path, map_location=self.device))
                    self.critic_1.eval()
                case "critic_2":
                    self.critic_2.load_state_dict(torch.load(path, map_location=self.device))
                    self.critic_2.eval()
                case _:
                    raise ValueError("Invalid network name.") 
        return True   


    def select_action(self, state: any, noise: bool = False) -> any:
        """
        Select an action based on the current state.

        Args:
            state (any): The current state.
            noise (bool): Whether to add exploration noise.

        Returns:
            any: The action to take.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            action = self.actor(state).cpu().numpy().flatten()

            if noise:
                action += np.random.normal(0, self.exploration_noise, size=action.shape)
                action = np.clip(action, -self.max_action, self.max_action)

            return action

    def train(self) -> float:
        """
        Perform a training step.

        Returns:
            float: The critic loss.
        """
        self.training = True
        self.total_it += 1

        if len(self.memory) < self.batch_size:
            return 0

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        # Compute critic losses
        critic_loss_1 = nn.MSELoss()(current_Q1, target_Q)
        critic_loss_2 = nn.MSELoss()(current_Q2, target_Q)

        # Optimize the first critic
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        # Optimize the second critic
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_target_1)
            self._soft_update(self.critic_2, self.critic_target_2)

            return critic_loss_1.item() + critic_loss_2.item()

        return 0

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """
        Perform a soft update of the target network parameters.

        Args:
            source (nn.Module): Source network.
            target (nn.Module): Target network.
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            ) 
