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
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        """Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Max number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start: Initial value of beta for importance sampling
            beta_frames: Number of frames over which to anneal beta to 1
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # Current frame number, used for beta annealing

        # Initialize buffers for storing transitions
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Initialize segment trees for efficient priority operations
        self.size = 0
        self.next_idx = 0

        # Segment trees
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a new experience to memory."""
        # Store transition
        self.states[self.next_idx] = state
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.next_states[self.next_idx] = next_state
        self.dones[self.next_idx] = done

        # Update priorities
        self.sum_tree[self.next_idx] = self.max_priority ** self.alpha
        self.min_tree[self.next_idx] = self.max_priority ** self.alpha

        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple:
        """Sample a batch of experiences."""
        # Calculate current beta for importance sampling
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1

        indices = self._sample_proportional(batch_size)
        weights = self._calculate_weights(indices, beta)

        # Convert to torch tensors and move to device
        states = torch.FloatTensor(self.states[indices]).to(device)
        actions = torch.FloatTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        weights = torch.FloatTensor(weights).to(device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities), f"Indices and priorities length mismatch: {len(indices)} vs {len(priorities)}"
        assert priorities.ndim == 1, f"Priorities should be 1D array, got shape {priorities.shape}"
        
        # Ensure all priorities are positive
        assert np.all(priorities > 0), "All priorities must be positive"
        
        for idx, priority in zip(indices, priorities):
            assert 0 <= idx < self.capacity, f"Index {idx} out of bounds"
            
            priority_alpha = float(priority ** self.alpha)
            self.sum_tree[idx] = priority_alpha
            self.min_tree[idx] = priority_alpha
            
            self.max_priority = max(self.max_priority, float(priority))

    def _sample_proportional(self, batch_size: int) -> np.ndarray:
        """Sample indices based on proportional prioritization."""
        indices = []
        total_priority = self.sum_tree.sum(0, self.size)

        for _ in range(batch_size):
            mass = np.random.random() * total_priority
            idx = self.sum_tree.find_prefixsum_idx(mass)
            indices.append(idx)

        return np.array(indices)

    def _calculate_weights(self, indices: np.ndarray, beta: float) -> np.ndarray:
        """Calculate importance sampling weights."""
        # Get min priority (avoid division by zero)
        min_priority = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_priority * self.size) ** (-beta)

        # Calculate weights
        weights = []
        for idx in indices:
            priority = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (priority * self.size) ** (-beta)
            weights.append(weight / max_weight)

        return np.array(weights)

    def __len__(self) -> int:
        """Return the current size of memory."""
        return self.size