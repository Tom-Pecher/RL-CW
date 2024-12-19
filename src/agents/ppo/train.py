"""
Training script for the PPO agent.
"""

from agent import Agent


class PPOAgent(Agent):
    def __init__(self,hardcore: bool, render: bool, env) -> None:
        self.env = env 
        pass
