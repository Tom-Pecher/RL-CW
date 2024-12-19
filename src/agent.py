from abc import abstractmethod
from gym.wrappers.record_video import RecordVideo


class Agent:
    def __init__(self,env) -> None:
        self.env = env 
        self.agent_name = "agent"

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def select_action(self,state):
        pass

    def record_agent(self, output_folder: str = "ouput") -> None:
        # Setup video recording
        record_env = RecordVideo(
            self.env,
            video_folder=output_folder,
            episode_trigger=lambda _: True,
            name_prefix=self.agent_name
        )

        # Run one episode
        state, _ = record_env.reset()
        episode_reward = 0

        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = record_env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            if done:
                break

        record_env.close()
