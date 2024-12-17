"""
Training script for the deep Q-learning agent.
"""

import torch
import os
import wandb

from bipedal_walker.environment import BipedalWalkerEnv
from agents.dqn.agent import DQNAgent
from gym.wrappers.record_video import RecordVideo

def train_agent(hardcore: bool, render: bool):
    """
    Runs an example agent with a single episode.
    """

    # Initialize WandB
    wandb.init(
        project="bipedal-walker",
        config={
            "algorithm": "DDPG",
            "environment": "BipedalWalker-v3",
            "hardcore": hardcore,
            "num_episodes": 1000,
        }
    )

    base_env = BipedalWalkerEnv(hardcore, render)

    # Create output dir for videos
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)

    episode_trigger_count = 100

    # Record a video every 200 episodes
    env = RecordVideo(
        base_env.env,
        video_dir,
        episode_trigger = lambda ep: ep % episode_trigger_count == 0,
        name_prefix="dqn"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env_info = base_env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']

    agent = DQNAgent(state_dim, action_dim, device)

    num_episodes = 1000
    best_reward = float('-inf')

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)

            # Train the agent
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train()
                episode_loss += loss

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break

        avg_loss = (episode_loss / steps) if steps > 0 else 0

        # Log metrics to WandB and print to console
        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "average_loss": avg_loss,
            "steps": steps,
            "buffer_size": len(agent.memory)
        })
        print(f"Log: Episode {episode + 1}/{num_episodes}, "
              f"Reward: {episode_reward:.2f}, "
              f"Avg loss: {avg_loss:.4f}, "
              f"Epsilon: {agent.epsilon:.4f}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_dir = "models"
            os.makedirs(os.path.join(model_dir, "dqn"), exist_ok=True)
            torch.save(agent.policy_net.state_dict(),
                        f"{model_dir}/dqn/best_policy.pth")
            torch.save(agent.target_net.state_dict(),
                        f"{model_dir}/dqn/best_target.pth")
            wandb.log({"best_reward": best_reward})


        # Save periodic checkpoints
        if (episode) % episode_trigger_count == 0:
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)

            checkpoint_path_policy = f"{model_dir}/dqn/ep_{episode}_policy.pth"
            checkpoint_path_target = f"{model_dir}/dqn/ep_{episode}_target.pth"

            # Save checkpoints
            torch.save(agent.policy_net.state_dict(), checkpoint_path_policy)
            torch.save(agent.target_net.state_dict(), checkpoint_path_target)

            # Log checkpoints to WandB
            wandb.save(checkpoint_path_policy)
            wandb.save(checkpoint_path_target)

            # Log videos to WandB
            wandb.log({
                "video": wandb.Video(
                    f"{video_dir}/dqn-episode-{episode}.mp4",
                    format="mp4"
                )
            })

    wandb.finish()
    env.close()
    return agent
