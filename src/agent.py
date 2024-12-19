from abc import ABC,abstractmethod
from gym.wrappers.record_video import RecordVideo


class Agent(ABC):

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def select_action(self,state):
        pass

    @abstractmethod
    def get_agent_name() -> str:
        raise NotImplementedError("Get agent name not implemented")

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
