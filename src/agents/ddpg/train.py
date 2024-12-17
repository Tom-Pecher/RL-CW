"""
Training script for the DDPG agent.
"""

from bipedal_walker.environment import BipedalWalkerEnv

def train_agent(hardcore: bool, render: bool):
    """
    Trains the DDPG agent.
    """
    env = BipedalWalkerEnv(hardcore, render)

    # TODO: Implement DDPG
    raise NotImplementedError
