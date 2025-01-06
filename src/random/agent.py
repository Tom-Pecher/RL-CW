
import numpy as np

class RandomAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float, device: any):
        """Initialize the random agent."""
        self.action_dim = action_dim
        self.max_action = max_action

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select a random action."""
        return np.random.uniform(-self.max_action, self.max_action, self.action_dim)

    def train(self) -> float:
        """No training for random agent."""
        return 0.0
