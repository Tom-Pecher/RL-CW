"""
A simple implementation of a replay buffer.
"""

from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """ Initialize the replay buffer. """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        """ Push a new transition into the replay buffer. """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """ Sample a batch of transitions from the replay buffer. """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        """ Get the number of transitions in the replay buffer. """
        return len(self.buffer)
