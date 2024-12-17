"""
Training script for the actor-critic agent.
"""


from bipedal_walker.environment import BipedalWalkerEnv

def train_agent(hardcore: bool, render: bool):
    """
    Trains the actor-critic agent.
    """
    env = BipedalWalkerEnv(hardcore, render)

    # TODO: Implement actor-critic
    raise NotImplementedError
