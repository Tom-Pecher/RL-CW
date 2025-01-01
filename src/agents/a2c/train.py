import torch
import os
import wandb
import torch.multiprocessing as mp

from bipedal_walker.environment import BipedalWalkerEnv
from agents.a2c.agent import A2CAgent

from gym.wrappers.record_video import RecordVideo

def train_agent(hardcore: bool, render: bool):
    """
    Trains the A2C agent using multiple parallel workers.
    """
    mp.set_start_method('spawn', force=True)

    num_workers = 4  # Number of parallel workers

    # Initialize WandB
    wandb.init(
        project="bipedal-walker",
        config={
            "algorithm": "A2C",
            "environment": "BipedalWalker-v3",
            "hardcore": hardcore,
            "num_workers": num_workers
        }
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create a sample environment to get dimensions
    env = BipedalWalkerEnv(hardcore, render)
    env_info = env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    max_action = float(env_info['action_high'][0])

    # Create A2C agent
    agent = A2CAgent(state_dim, action_dim, max_action, device, hardcore, num_workers)

    print("Starting training with", num_workers, "workers...")

    try:
        agent.train()

    except KeyboardInterrupt:
        print("Training interrupted!")

    wandb.finish()
    env.close()
    return agent
