"""
Training script for the DDPG agent.
"""

import torch
import os
import wandb

from bipedal_walker.environment import BipedalWalkerEnv
from agents.ddpg.agent import DDPGAgent

from gym.wrappers.record_video import RecordVideo

def train_agent(hardcore: bool, render: bool):
    """
    Trains the DDPG agent.
    """

    num_episodes = 1000

    # Initialize WandB
    wandb.init(
        project="bipedal-walker",
        config={
            "algorithm": "DDPG",
            "environment": "BipedalWalker-v3",
            "hardcore": hardcore,
            "num_episodes": num_episodes,
        }
    )

    # Create environment
    base_env = BipedalWalkerEnv(hardcore, render)

    # Create output dir for videos
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)

    episode_trigger_count = 100

    # Setup video recording
    env = RecordVideo(
        base_env.env,
        video_dir,
        episode_trigger = lambda ep: (ep % episode_trigger_count == 0) or (ep == num_episodes - 1),
        name_prefix="ddpg"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env_info = base_env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    max_action = float(env_info['action_high'][0])

    agent = DDPGAgent(state_dim, action_dim, max_action, device)

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

            agent.memory.push(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.batch_size:
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
            os.makedirs(os.path.join(model_dir, "ddpg"), exist_ok=True)
            torch.save(agent.actor.state_dict(),
                        f"{model_dir}/ddpg/best_actor.pth")
            torch.save(agent.critic.state_dict(),
                        f"{model_dir}/ddpg/best_critic.pth")
            wandb.log({"best_reward": best_reward})

        # Save periodic checkpoints
        if (episode % episode_trigger_count == 0) or (episode == num_episodes - 1):
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)

            checkpoint_path_actor = f"{model_dir}/ddpg/ep_{episode}_actor.pth"
            checkpoint_path_critic = f"{model_dir}/ddpg/ep_{episode}_critic.pth"

            # Save checkpoints
            torch.save(agent.actor.state_dict(), checkpoint_path_actor)
            torch.save(agent.critic.state_dict(), checkpoint_path_critic)

            # Log checkpoints to WandB
            wandb.save(checkpoint_path_actor)
            wandb.save(checkpoint_path_critic)

            # Log videos to WandB
            wandb.log({
                "video": wandb.Video(
                    f"{video_dir}/ddpg-episode-{episode}.mp4",
                    format="mp4"
                )
            })

    wandb.finish()
    env.close()
    return agent
