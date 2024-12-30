"""
A simple implementation of a replay buffer.
"""

import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        # Ensure inputs are numpy arrays
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device) -> tuple:
        transitions = random.sample(self.buffer, batch_size)

        # Convert to numpy arrays
        batch = map(np.stack, zip(*transitions))
        state, action, reward, next_state, done = batch

        # Convert numpy arrays to tensors
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).reshape(-1, 1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).reshape(-1, 1).to(device)
        )

    def get_minibatches(self,batch_size: int,device: torch.device) -> list[tuple]:

        batches = []

        #Shuffle the buffer
        buffer_list = list(self.buffer)
        random.shuffle(buffer_list)
        for i in range(0,len(self.buffer),batch_size):
            
            #break up shuffled data into batches

            batch = buffer_list[i:i+batch_size]
            batch = map(np.stack, zip(*batch))
            state, action, reward, next_state, done = batch

            # append batch to batches
            batches.append((
                torch.FloatTensor(state).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor(reward).reshape(-1, 1).to(device),
                torch.FloatTensor(next_state).to(device),
                torch.FloatTensor(done).reshape(-1, 1).to(device)
            )) 

        return batches
        
    def __len__(self) -> int:
        return len(self.buffer)

