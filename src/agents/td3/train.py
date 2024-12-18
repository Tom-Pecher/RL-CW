"""
Training script for the TD3 agent.
"""

import torch
import os
import wandb

from bipedal_walker.environment import BipedalWalkerEnv
from agents.td3.agent import TD3Agent

from gym.wrappers.record_video import RecordVideo


def train_agent(hardcore: bool, render: bool):
    """
    Trains the TD3 agent.
    """

    num_episodes = 1000

    # Initialize WandB
    wandb.init(
        project="bipedal-walker-td3",
        config={
            "algorithm": "TD3",
            "environment": "BipedalWalker-v3",
            "hardcore": hardcore,
            "num_episodes": num_episodes,
        }
    )

    # Create environment
    base_env = BipedalWalkerEnv(hardcore, render)

    # Create output dir for videos
    video_dir = "videos/td3"
    os.makedirs(video_dir, exist_ok=True)

    episode_trigger_count = 100

    # Setup video recording
    env = RecordVideo(
        base_env.env,
        video_dir,
        episode_trigger=lambda ep: (ep % episode_trigger_count == 0) or (ep == num_episodes - 1),
        name_prefix="video"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env_info = base_env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    max_action = float(env_info['action_high'][0])

    agent = TD3Agent(state_dim, action_dim, max_action, device)

    best_reward = float('-inf')

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0

        while True:
            action = agent.select_action(state, noise=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)

            loss = agent.train()
            episode_loss += loss

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break

        avg_loss = episode_loss / steps if steps > 0 else 0

        # Log metrics to WandB and print to console
        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "average_loss": avg_loss,
            "steps": steps,
            "buffer_size": len(agent.memory)
        })
        print(f"Episode {episode + 1}/{num_episodes}, "
              f"Reward: {episode_reward:.2f}, "
              f"Avg Loss: {avg_loss:.4f}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_dir = "models"
            os.makedirs(os.path.join(model_dir, "td3"), exist_ok=True)
            torch.save(agent.actor.state_dict(),
                       f"{model_dir}/td3/best_actor.pth")
            torch.save(agent.critic_1.state_dict(),
                       f"{model_dir}/td3/best_critic_1.pth")
            torch.save(agent.critic_2.state_dict(),
                       f"{model_dir}/td3/best_critic_2.pth")
            wandb.log({"best_reward": best_reward})

        # Save periodic checkpoints
        if (episode % episode_trigger_count == 0) or (episode == num_episodes - 1):
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)

            checkpoint_path_actor = f"{model_dir}/td3/ep_{episode}_actor.pth"
            checkpoint_path_critic1 = f"{model_dir}/td3/ep_{episode}_critic1.pth"
            checkpoint_path_critic2 = f"{model_dir}/td3/ep_{episode}_critic2.pth"

            # Save checkpoints
            torch.save(agent.actor.state_dict(), checkpoint_path_actor)
            torch.save(agent.critic_1.state_dict(), checkpoint_path_critic1)
            torch.save(agent.critic_2.state_dict(), checkpoint_path_critic2)

            # Log checkpoints to WandB
            wandb.save(checkpoint_path_actor)
            wandb.save(checkpoint_path_critic1)
            wandb.save(checkpoint_path_critic2)

            # Log videos to WandB
            wandb.log({
                "video": wandb.Video(
                    f"{video_dir}/video-episode-{episode}.mp4",
                    format="mp4"
                )
            })

    wandb.finish()
    env.close()
    return agent 