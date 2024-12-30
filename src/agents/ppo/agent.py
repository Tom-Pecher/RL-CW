"""
Training script for the PPO agent.
"""

from agents.common.value_net import ValueNetwork
from agent import Agent
from typing import Dict
from agents.common.replay_buffer import ReplayBuffer
from agents.common.actor import Actor
import torch
import torch.optim as optim
import numpy as np

class PPOAgent(Agent):
    def __init__(self,env_info: dict, device: torch.device,epsilon: float = 0.1, gamma: float = 0.99, c1: float = 0.0, c2: float = 0.5, model_path: str | None = None) -> None:
        """
            Initializes the agent taking into account hyperparams
        """
        self.training = False
        #Defining the hyperparams
        self.epsilon = epsilon 
        self.gamma = gamma
        self.gae_param = 0.95
        self.batch_size = 64
        self.policy_lr = 3e-4
        self.value_lr = 3e-4
        self.exploration_noise = 0.1
        self.num_epoch = 2

        # Initializes agent
        state_dim = env_info['observation_dim']
        action_dim = env_info['action_dim']
        self.max_action = float(env_info['action_high'][0])
        self.policy_net = Actor(state_dim, action_dim, self.max_action).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.value_optimizer  = optim.Adam(self.value_net.parameters(),lr=self.value_lr)
        self.device = device
    
    @staticmethod
    def get_agent_name() -> str:
        return "ppo"
   
    def load_agent(self, model_path: str | Dict[str, str]) -> bool:
        
        if isinstance(model_path,str):
            # It might be worth try catching it 
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policy_net.eval()
            return True 

        if not model_path: # for some reason this checks if a dict is empty  
            return False

        for path, path_type in model_path: 
            match path_type:
                case "policy":
                    self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
                    self.policy_net.eval()
                case "value":
                    self.value_net.load_state_dict(torch.load(path, map_location=self.device))
                    self.value_net.eval()
                case _:
                    raise ValueError("Invalid network name.") 
        return True



    def select_action(self, state):
        """
        Select an action based on the current state.

        Args:
            state (any): The current state.

        Returns:
            any: The action to take.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            action = self.policy_net(state)

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

    def compute_gae(self,states,rewards,next_states, terminated):
        

        # initializing the state values 
        state_vals = self.value_net(torch.tensor(np.asarray(states),dtype=torch.float32,device=self.device))
        state_vals = list(state_vals.detach().cpu().numpy().flatten()) # normal amout of conversions lol

        # if robot is not dead add final state to values otherwise add a 0 
        state_vals.append(0 if terminated else self.value_net(torch.tensor(next_states[-1],dtype=torch.float32,device=self.device)).item())

        # compute deltas  
        deltas = []
        for x in range(len(state_vals) - 1):
            delta = rewards[x] + self.gamma * state_vals[x + 1] - state_vals[x] 
            deltas.append(delta)

        # calculate advantages
        advantages = []
        advantage = 0
        gae_gamma = self.gae_param * self.gamma
        for delta in reversed(deltas):
            advantage = delta + gae_gamma * advantage
            advantages.append(advantage)
        advantages = list(reversed(advantages))
        
        # calculate reward to go 
        reward_to_go = []
        current_reward = 0 
        for reward in  rewards:
            reward_to_go.append(current_reward)
            current_reward = reward + current_reward * self.gamma
        reward_to_go = list(reversed(reward_to_go))
        return advantages, reward_to_go

    
    def train_value(self, states, rewards_to_go) -> float:
        self.value_optimizer.zero_grad()
        vals = self.value_net(states)
        value_loss = torch.mean((vals - rewards_to_go) ** 2)
        value_loss.backward()
        self.value_optimizer.step()
        return value_loss.item()

    def train_policy(self,states,actions, advantages) -> float:
        """
            Trains the policy network 
        """

        self.policy_optimizer.zero_grad() 
        r_t = self.policy_net(states) / (actions + 1e-6)
        r_t_clip = r_t.clamp(
            1 - self.epsilon, 1 + self.epsilon
        )
        loss =  - torch.min(r_t * advantages, r_t_clip * advantages).mean() # we maximes it so the negative sign
        loss.backward() 
        self.policy_optimizer.step()
        return loss.item()
