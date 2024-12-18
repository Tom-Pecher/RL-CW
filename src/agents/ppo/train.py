"""
Training script for the PPO agent.
"""

from bipedal_walker.environment import BipedalWalkerEnv

def train_agent(hardcore: bool, render: bool):
    """
    Trains the PPO agent.
    """
    env = BipedalWalkerEnv(hardcore, render)

    # TODO: Implement PPO
    raise NotImplementedError
