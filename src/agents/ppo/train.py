"""
Training script for the PPO agent.
"""

import torch
import os
import wandb

from bipedal_walker.environment import BipedalWalkerEnv
from agents.ppo.agent import PPOAgent

from gym.wrappers.record_video import RecordVideo


def train_agent(hardcore: bool, render: bool):
    """
    Trains the PPO agent.
    """
    num_episodes = 1000

    # Initialize WandB
    wandb.init(
        project="bipedal-walker",
        config={
            "algorithm": "PPO",
            "environment": "BipedalWalker-v3",
            "hardcore": hardcore,
            "num_episodes": num_episodes,
        }
    )

    # Create environment
    base_env = BipedalWalkerEnv(hardcore, render)

    # Create output dir for videos
    video_dir = "videos/ppo"
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

    agent = PPOAgent(state_dim, action_dim, max_action, device)

    best_reward = float('-inf')

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_policy_loss = 0
        episode_value_loss = 0
        episode_entropy = 0
        steps = 0

        while True:
            # Select action and get value estimate
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)

            state = next_state
            episode_reward += reward
            steps += 1

            # Train if we have collected enough steps
            if len(agent.states) >= agent.trajectory_size or done:
                policy_loss, value_loss, entropy = agent.train()
                episode_policy_loss += policy_loss
                episode_value_loss += value_loss
                episode_entropy += entropy

            if done:
                break

        avg_policy_loss = episode_policy_loss / steps if steps > 0 else 0
        avg_value_loss = episode_value_loss / steps if steps > 0 else 0
        avg_entropy = episode_entropy / steps if steps > 0 else 0

        # Log metrics to WandB and print to console
        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "steps": steps
        })
        print(f"Episode {episode + 1}/{num_episodes}, "
              f"Reward: {episode_reward:.2f}, "
              f"Policy Loss: {avg_policy_loss:.4f}, "
              f"Value Loss: {avg_value_loss:.4f}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_dir = "models"
            os.makedirs(os.path.join(model_dir, "ppo"), exist_ok=True)
            torch.save(agent.policy.state_dict(),
                      f"{model_dir}/ppo/best_policy.pth")
            torch.save(agent.value.state_dict(),
                      f"{model_dir}/ppo/best_value.pth")
            wandb.log({"best_reward": best_reward})

        # Save periodic checkpoints
        if (episode % episode_trigger_count == 0) or (episode == num_episodes - 1):
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)

            checkpoint_path_policy = f"{model_dir}/ppo/ep_{episode}_policy.pth"
            checkpoint_path_value = f"{model_dir}/ppo/ep_{episode}_value.pth"

            # Save checkpoints
            torch.save(agent.policy.state_dict(), checkpoint_path_policy)
            torch.save(agent.value.state_dict(), checkpoint_path_value)

            # Log checkpoints to WandB
            wandb.save(checkpoint_path_policy)
            wandb.save(checkpoint_path_value)

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
