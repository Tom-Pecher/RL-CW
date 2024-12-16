"""
This is a simple example of training a model.
This runs a single episode.
"""

from bipedal_walker.environment import BipedalWalkerEnv

def train_agent(hardcore: bool, render: bool):
    """
    Runs an example agent with a single episode.
    """
    env = BipedalWalkerEnv(hardcore, render)

    env_info = env.get_env_info()
    print("Environment info:", env_info)

    # Run a simple episode
    observation, info = env.reset()
    done = False

    total_reward = 0

    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        done = terminated or truncated

        env.render()

    print(f"Episode finished with total reward: {total_reward}")
    print("Episode info:", info)
    env.close()
