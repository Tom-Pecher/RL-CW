import torch
import os

from bipedal_walker.environment import BipedalWalkerEnv
from agents.a2c.agent import A2CAgent

from gym.wrappers.record_video import RecordVideo

def record_agent(model_path: str, hardcore: bool = False) -> None:
    # Create environment
    base_env = BipedalWalkerEnv(hardcore=hardcore, render=False)

    # Setup video recording
    env = RecordVideo(
        base_env.env,
        video_folder="output",
        episode_trigger=lambda _: True,
        name_prefix="a2c"
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get environment info
    env_info = base_env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    max_action = float(env_info['action_high'][0])

    # Create agent and load model
    agent = A2CAgent(state_dim, action_dim, max_action, device)
    agent.shared_actor.load_state_dict(torch.load(model_path, map_location=device))
    agent.shared_actor.eval()

    # Run one episode
    state, _ = env.reset()
    episode_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        episode_reward += reward

        if done:
            break

    env.close()