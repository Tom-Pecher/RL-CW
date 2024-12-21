"""
Prioritized Experience Replay Buffer implementation.
"""

import numpy as np
import torch
from typing import Tuple
from .segment_tree import MinSegmentTree, SumSegmentTree

class PrioritizedReplayBuffer:
    def __init__(
        self, 
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ) -> None:
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum size of buffer
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            alpha: How much prioritization is used (0 = no prioritization, 1 = full prioritization)
            beta: Importance sampling correction factor
            beta_increment: Increment for beta over time
            epsilon: Small constant to prevent zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.pos = 0
        self.size = 0
        
        # Create segment trees for sum and minimum
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
            
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0
        
        # Storage for transitions with proper dimensions
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(
        self, 
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition and priority."""
        # Store transition
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        # Store max priority for new transition
        self.sum_tree[self.pos] = self.max_priority ** self.alpha
        self.min_tree[self.pos] = self.max_priority ** self.alpha
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple:
        """Sample a batch of transitions."""
        indices = self._sample_proportional(batch_size)
        
        # Calculate importance sampling weights
        weights = []
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-self.beta)
        
        for idx in indices:
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * self.size) ** (-self.beta)
            weights.append(weight / max_weight)
            
        weights = np.array(weights, dtype=np.float32)
        
        # Get batch of transitions
        states = torch.FloatTensor(self.states[indices]).to(device)
        actions = torch.FloatTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities of sampled transitions."""
        priorities = np.abs(priorities) + self.epsilon
        
        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size: int) -> np.ndarray:
        """Sample indices based on proportional prioritization."""
        indices = []
        total = self.sum_tree.sum()
        
        for _ in range(batch_size):
            mass = np.random.random() * total
            idx = self.sum_tree.find_prefixsum_idx(mass)
            indices.append(idx)
            
        return np.array(indices)

    def __len__(self) -> int:
        return self.size