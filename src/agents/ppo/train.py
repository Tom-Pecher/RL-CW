"""
Training script for the PPO agent.
"""

from bipedal_walker.environment import BipedalWalkerEnv

def train_agent(hardcore: bool, render: bool):
    """
    Runs an example agent with a single episode.
    """
    env = BipedalWalkerEnv(hardcore, render)

    # TODO: Implement PPO
    raise NotImplementedError
