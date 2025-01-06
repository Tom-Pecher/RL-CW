
import torch
import os
import wandb

from bipedal_walker.environment import BipedalWalkerEnv
from agents.random.agent import RandomAgent

from gym.wrappers.record_video import RecordVideo

def train_agent(hardcore: bool, render: bool):
    """Trains (runs) the random agent."""

    num_episodes = 1000

    # Initialize WandB
    wandb.init(
        project="bipedal-walker",
        config={
            "algorithm": "Random",
            "environment": "BipedalWalker-v3",
            "hardcore": hardcore,
            "num_episodes": num_episodes,
        }
    )

    # Create environment
    base_env = BipedalWalkerEnv(hardcore, render)

    # Create output dir for videos
    video_dir = "videos/random"
    os.makedirs(video_dir, exist_ok=True)

    # Setup video recording
    env = RecordVideo(
        base_env.env,
        video_dir,
        episode_trigger=lambda ep: ep % 100 == 0,
        name_prefix="video"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env_info = base_env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    max_action = float(env_info['action_high'][0])

    agent = RandomAgent(state_dim, action_dim, max_action, device)
    best_reward = float('-inf')

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break

        # Log metrics to WandB and print to console
        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "steps": steps,
        })
        print(f"Episode {episode + 1}/{num_episodes}, "
              f"Reward: {episode_reward:.2f}, "
              f"Steps: {steps}")

        # Track best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            wandb.log({"best_reward": best_reward})

    wandb.finish()
    env.close()
    return agent
