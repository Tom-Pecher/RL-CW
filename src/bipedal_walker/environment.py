import gym
import numpy as np
from typing import Tuple, Dict, Any
from gym import spaces


class BipedalWalkerEnv:
    """
    A wrapper interface for OpenAI Gym's BipedalWalker environment.
    """

    def __init__(self, render_mode: str = None):
        """
        Initialize the environment.

        Args:
            render_mode(str): The rendering mode to use. ('human', 'rgb_array', None)
        """
        self.env = gym.make('BipedalWalker-v3', render_mode=render_mode)

        # Store environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Track episode stats
        self.current_step  = 0
        self.total_reward  = 0
        self.episode_count = 0


    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return the initial state.

        Returns:
            Tuple[np.ndarray, Dict]: The initial state and the initial info.
        """
        self.current_step = 0
        self.total_reward = 0
        return self.env.reset()


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Takes a step in the environment.

        Args:
            action(np.ndarray): The action to take.

        Returns:
            observation (np.ndarray): Current observation
            reward (float): Reward received from the step
            terminated (bool): Whether the episode has terminated
            truncated (bool): Whether the episode was truncated
            info (dict): Additional information
        """
        self.current_step += 1

        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Take the step
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.total_reward += reward

        if terminated or truncated:
            self.episode_count += 1
            info.update({
                'episode': {
                    'r': self.total_reward,
                    'l': self.current_step,
                }
            })

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the environment."""
        return self.env.render()

    def close(self):
        """Closes the environment."""
        return self.env.close()

    def seed(self, seed: int):
        """
        Sets the seed for the environment.

        Args:
            seed (int): The seed to use.
        """
        return self.env.seed(seed)

    def sample_action(self) -> np.ndarray:
        """
        Samples an action from the action space.

        Returns:
            np.ndarray: The randomly sampled action.
        """
        return self.env.action_space.sample()

    @property
    def get_observation_space(self) -> spaces.Box:
        """Returns the observation space."""
        return self.observation_space

    @property
    def get_action_space(self) -> spaces.Box:
        """Returns the action space."""
        return self.action_space

    @property
    def get_current_step(self) -> int:
        """Returns the current step."""
        return self.current_step

    @property
    def get_total_reward(self) -> float:
        """Returns the total reward."""
        return self.total_reward

    @property
    def get_episode_count(self) -> int:
        """Returns the episode count."""
        return self.episode_count
