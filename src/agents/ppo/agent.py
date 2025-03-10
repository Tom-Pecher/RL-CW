"""
Training script for the PPO agent.
"""

from agents.common.policy_network import PolicyNetwork
from agents.common.value_network import ValueNetwork
from typing import Dict
import torch
import torch.optim as optim
import numpy as np

class PPOAgent:
    def __init__(self,env_info: dict, device: torch.device,epsilon: float = 0.2, gamma: float = 0.99, c1: float = 1.0, c2: float = 0.01, model_path: str | None = None) -> None:
        """
            Initializes the agent taking into account hyperparams
        """
        self.training = False
        #Defining the hyperparams
        self.epsilon = epsilon 
        self.gamma = gamma
        self.c1 = c1
        self.c2 = c2
        self.gae_param = 0.95
        self.batch_size = 1024
        self.lr = 3e-4
        self.num_epoch = 5 

        # Initializes agent
        state_dim = env_info['observation_dim']
        action_dim = env_info['action_dim']
        max_action = float(env_info['action_high'][0])
        
        UPPER_BOUND = 2
        LOWER_BOUND = -20
        self.policy_net = PolicyNetwork(state_dim, action_dim, max_action, [64, 64], UPPER_BOUND, LOWER_BOUND).to(device)
        self.value_net = ValueNetwork(state_dim,[64,64]).to(device)


        self.optimizer = optim.Adam(
            list(self.value_net.parameters()) + 
            list(self.policy_net.parameters()), 
            lr=self.lr
        )
       
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
            dist = self.policy_net.get_distribution(state)
            if self.training:
                action = dist.sample()
            else:
                print("hello")
                action = dist.mean()
        return action.cpu().numpy().flatten(), dist.log_prob(action).sum(-1).cpu().item()

            


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

    
    def train(self,states,actions,log_probs, advantages,rewards_to_go) -> float:
        """
            Trains the policy network 
        """

        self.optimizer.zero_grad()
        new_dist = self.policy_net.get_distribution(states)
        new_log_prob = new_dist.log_prob(actions).sum(1)
        r_t = torch.exp( new_log_prob - log_probs)
        r_t_clip = r_t.clamp(
            1 - self.epsilon, 1 + self.epsilon
        )
        loss_clip =  torch.min(r_t * advantages, r_t_clip * advantages).mean() # we maximes it so the negative sign
        vals = self.value_net(states)
        loss_val = torch.mean((vals - rewards_to_go) ** 2)
        entropy_loss = new_dist.entropy().mean()
        loss = - loss_clip + self.c1 * loss_val - self.c2 * entropy_loss 
        loss.backward() 
        self.optimizer.step()
        return loss.item()
