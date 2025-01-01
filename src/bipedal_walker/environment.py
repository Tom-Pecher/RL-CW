import gym
import numpy as np
from typing import Tuple, Dict, Any
from gym import spaces


class BipedalWalkerEnv:
    """
    A wrapper interface for OpenAI Gym's BipedalWalker environment.
    """

    def __init__(self, hardcore: bool, render: bool):
        """
        Initialize the environment.

        Args:
            hardcore(bool): Whether to use the hardcore version of the environment.
            render_mode(bool): Whether to render the environment.
        """
        self.env = gym.make(
            'BipedalWalker-v3',
            hardcore=hardcore,
            render_mode='human' if render else 'rgb_array'
        )

        # Store environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Track episode stats
        self.current_step  = 0
        self.total_reward  = 0
        self.episode_count = 0

        # Store observation bounds for normalization
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high
        self.obs_range = self.obs_high - self.obs_low

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize observation to [-1, 1] range using stored bounds.

        Args:
            observation(np.ndarray): Raw observation from environment.

        Returns:
            np.ndarray: Normalized observation.
        """
        return 2.0 * (observation - self.obs_low) / self.obs_range - 1.0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return the initial state.

        Returns:
            Tuple[np.ndarray, Dict]: The initial state and the initial info.
        """
        self.current_step = 0
        self.total_reward = 0
        observation, info = self.env.reset()
        return self._normalize_observation(observation), info

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

        # Normalize the observation
        normalized_observation = self._normalize_observation(observation)

        self.total_reward += reward

        if terminated or truncated:
            self.episode_count += 1
            info.update({
                'episode': {
                    'r': self.total_reward,
                    'l': self.current_step,
                }
            })

        return normalized_observation, reward, terminated, truncated, info


    def render(self):
        """Renders the environment."""
        return self.env.render()


    def close(self):
        """Closes the environment."""
        return self.env.close()


    def sample_action(self) -> np.ndarray:
        """
        Samples an action from the action space.

        Returns:
            np.ndarray: The randomly sampled action.
        """
        return self.env.action_space.sample()


    def get_env_info(self) -> Dict[str, Any]:
        """
        Get information about the environment.

        Returns:
            dict: Environment information including dimensions and ranges
        """
        return {
            'observation_dim' : self.observation_space.shape[0],
            'action_dim'      : self.action_space.shape[0],
            'action_low'      : self.action_space.low,
            'action_high'     : self.action_space.high,
            'observation_low' : self.observation_space.low,
            'observation_high': self.observation_space.high
        }


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
