from abc import ABC,abstractmethod
from typing import Dict
import torch

class Agent(ABC):

    @abstractmethod
    def select_action(self,state: any) -> any:
        raise NotImplementedError("select_action not implemented")

    @staticmethod
    @abstractmethod
    def get_agent_name() -> str:
        raise NotImplementedError("get_agent_name not implemented")

    @abstractmethod
    def load_agent(self,model_path: str | Dict[str,str]) -> bool:
        raise NotImplementedError("load_actor is not implemented")

    def record_agent(self, video_env) -> None:
        """
            In order for this to work efficient with neural networks the option of 
            Initialization with pretrained model weight should be used included
        """
        # Run one episode
        state, _ = video_env.reset()
        episode_reward = 0

        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            if done:
                break

        video_env.close()
