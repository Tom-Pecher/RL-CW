"""
Agent for the DDPG (Deep Deterministic Policy Gradient) RL.
"""


from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from agents.common.actor import Actor
from agents.common.critic import Critic
from agents.common.replay_buffer import ReplayBuffer
from agent import Agent

class DDPGAgent(Agent):
    def __init__(self,env_info: Dict[str,any], device: torch.device) -> None:
        """ Initialize the DDPG agent. """
        state_dim = env_info['observation_dim']
        action_dim = env_info['action_dim']


        self.device = device
        self.max_action = float(env_info['action_high'][0])

        self.training = False

        # Training hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.exploration_noise = 0.1
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

        # Actor
        self.actor = Actor(state_dim, action_dim, self.max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.actor_lr)

        self.memory = ReplayBuffer(1000000)
        self.total_it = 0

    @staticmethod
    def get_agent_name() -> str:
        return "ddpg"
    
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
                case "critic":
                    self.critic.load_state_dict(torch.load(path, map_location=self.device))
                    self.critic.eval()
                case _:
                    raise ValueError("Invalid network name.") 
        return True

    def select_action(self, state: any) -> any:
        """
        Select an action based on the current state.

        Args:
            state (any): The current state.

        Returns:
            any: The action to take.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            action = self.actor(state)

            if self.training:
                noise = torch.normal(
                    0,
                    self.exploration_noise,
                    size=action.shape,
                    device=self.device
                )
                action = torch.clamp(
                    action + noise,
                    -self.max_action,
                    self.max_action
                )

            return action.cpu().numpy().flatten()


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

        # Compute target Q value
        target_Q = self.critic_target(next_states, self.actor_target(next_states))
        target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Get current Q estimate
        current_Q = self.critic(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item()
