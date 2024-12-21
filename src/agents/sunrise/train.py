"""
Training script for the Sunrise agent.
"""

import torch
import os
import wandb

from bipedal_walker.environment import BipedalWalkerEnv
from agents.sunrise.agent import SUNRISEAgent

from gym.wrappers.record_video import RecordVideo


def train_agent(hardcore: bool, render: bool):
    """
    Trains the Sunrise agent.
    """

    num_episodes = 1000

    # Initialize WandB
    wandb.init(
        project="bipedal-walker",
        config={
            "algorithm": "Sunrise",
            "environment": "BipedalWalker-v3",
            "hardcore": hardcore,
            "num_episodes": num_episodes,
        }
    )

    # Create environment
    base_env = BipedalWalkerEnv(hardcore, render)

    # Create output dir for videos
    video_dir = "videos/sunrise"
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

    agent = SUNRISEAgent(state_dim, action_dim, max_action, device)

    best_reward = float('-inf')

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0

        while True:
            action = agent.select_action(state, evaluate=False)
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
            os.makedirs(os.path.join(model_dir, "sunrise"), exist_ok=True)

            # Save actor
            torch.save(agent.actor.state_dict(),
                      f"{model_dir}/sunrise/best_actor.pth")

            # Save all critics in the ensemble
            for i, critic in enumerate(agent.critics):
                torch.save(critic.state_dict(),
                          f"{model_dir}/sunrise/best_critic_{i}.pth")

            wandb.log({"best_reward": best_reward})

        # Save periodic checkpoints
        if (episode % episode_trigger_count == 0) or (episode == num_episodes - 1):
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)

            # Save actor checkpoint
            checkpoint_path_actor = f"{model_dir}/sunrise/ep_{episode}_actor.pth"
            torch.save(agent.actor.state_dict(), checkpoint_path_actor)

            # Save all critics checkpoints
            checkpoint_paths_critics = []
            for i, critic in enumerate(agent.critics):
                path = f"{model_dir}/sunrise/ep_{episode}_critic_{i}.pth"
                torch.save(critic.state_dict(), path)
                checkpoint_paths_critics.append(path)

            # Log checkpoints to WandB
            wandb.save(checkpoint_path_actor)
            for path in checkpoint_paths_critics:
                wandb.save(path)

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
