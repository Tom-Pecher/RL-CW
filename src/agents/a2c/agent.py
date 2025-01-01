import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import threading

from agents.common.actor import Actor
from agents.common.critic import Critic

from bipedal_walker.environment import BipedalWalkerEnv

# A2C Worker (local networks)
class A2CWorker(mp.Process):
    def __init__(self, worker_id, shared_actor, shared_critic, state_dim, action_dim, max_action, device, hardcore):
        super(A2CWorker, self).__init__()
        self.worker_id = worker_id
        self.device = device
        self.max_action = max_action
        self.hardcore = hardcore
        self.episode = 0

        # Create local actor and critic
        self.local_actor = Actor(state_dim, action_dim, max_action).to(device)
        self.local_critic = Critic(state_dim, action_dim).to(device)

        # Store shared networks
        self.shared_actor = shared_actor
        self.shared_critic = shared_critic

        # Hyperparameters
        self.gamma = 0.99
        self.lr = 3e-4
        self.entropy_coef = 0.01
        self.max_grad_norm = 40.0

        # Create optimizers for the shared networks:
        self.optimizer_actor = optim.Adam(self.shared_actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.shared_critic.parameters(), lr=self.lr)

    def create_env(self):
        """Create environment for this worker"""
        return BipedalWalkerEnv(hardcore=self.hardcore, render=False)

    def sync_with_shared(self):
        """Sync local networks with shared networks"""
        self.local_actor.load_state_dict(self.shared_actor.state_dict())
        self.local_critic.load_state_dict(self.shared_critic.state_dict())

    def select_action(self, state):
        """Select action using local actor network"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.local_actor(state)
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -self.max_action, self.max_action)
            return action.cpu().numpy().flatten()

    def compute_returns(self, rewards, dones, next_value):
        """Compute returns for each timestep"""
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def update_shared_networks(self, loss):
        """Update shared networks with gradients from local networks"""
        # Calculate gradients using local networks
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.max_grad_norm)
        
        # Ensure thread-safe updates to shared networks
        with threading.Lock():
            for shared_param, local_param in zip(self.shared_actor.parameters(), self.local_actor.parameters()):
                if shared_param.grad is None:
                    shared_param.grad = local_param.grad.clone()
                else:
                    shared_param.grad += local_param.grad.clone()
                    
            for shared_param, local_param in zip(self.shared_critic.parameters(), self.local_critic.parameters()):
                if shared_param.grad is None:
                    shared_param.grad = local_param.grad.clone()
                else:
                    shared_param.grad += local_param.grad.clone()
                    
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()

    def run(self):
        env = self.create_env()
        
        while True:
            # Sync with shared networks at the start of each episode
            self.sync_with_shared()
            self.episode += 1
            
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            # Storage for trajectory
            states, actions, rewards, dones = [], [], [], []
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                
                # If episode is done or enough steps have elapsed, perform update
                if done or len(states) >= 5:
                    # Convert to tensors
                    states_t = torch.FloatTensor(states).to(self.device)
                    actions_t = torch.FloatTensor(actions).to(self.device)
                    
                    # Compute returns
                    with torch.no_grad():
                        next_value = 0 if done else self.local_critic(
                            torch.FloatTensor([next_state]).to(self.device),
                            torch.FloatTensor([self.select_action(next_state)]).to(self.device)
                        ).item()
                    returns = self.compute_returns(rewards, dones, next_value)
                    
                    # Compute advantages
                    values = self.local_critic(states_t, actions_t)
                    advantages = returns - values.detach()
                    
                    # Compute actor (policy) loss
                    new_actions = self.local_actor(states_t)
                    critic_values = self.local_critic(states_t, new_actions)
                    actor_loss = -critic_values.mean()
                    
                    # Compute critic loss
                    critic_loss = ((returns - values) ** 2).mean()
                    
                    # Total loss
                    total_loss = actor_loss + critic_loss
                    
                    # Update networks
                    self.update_shared_networks(total_loss)
                    
                    # Clear trajectory storage
                    states, actions, rewards, dones = [], [], [], []
            
            # Log episode results
            print(f"Worker {self.worker_id}: Episode {self.episode} reward: {episode_reward}")

class A2CAgent:
    def __init__(self, state_dim, action_dim, max_action, device, hardcore, num_workers=4):
        self.device = device
        self.max_action = max_action
        self.num_workers = num_workers
        self.hardcore = hardcore

        # Shared networks
        self.shared_actor = Actor(state_dim, action_dim, max_action).to(device)
        self.shared_actor.share_memory()

        self.shared_critic = Critic(state_dim, action_dim).to(device)
        self.shared_critic.share_memory()

        # Initialize weights
        for m in self.shared_actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

        for m in self.shared_critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Create workers
        self.workers = []
        for i in range(num_workers):
            worker = A2CWorker(i, self.shared_actor, self.shared_critic,
                             state_dim, action_dim, max_action, device, hardcore)
            self.workers.append(worker)

    def train(self):
        """Start all workers"""
        for worker in self.workers:
            worker.start()

        for worker in self.workers:
            worker.join()
