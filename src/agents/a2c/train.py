"""
Training script for the A2C agent.
"""

import torch
import os
import wandb

from bipedal_walker.environment import BipedalWalkerEnv
from agents.a2c.agent import A2CAgent

from gym.wrappers.record_video import RecordVideo

def train_agent(hardcore: bool, render: bool):
    """
    Trains the A2C agent.
    """

    num_episodes = 1000

    # Initialize WandB
    wandb.init(
        project="bipedal-walker",
        config={
            "algorithm": "A2C",
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
        episode_trigger=lambda ep: (ep % episode_trigger_count == 0) or (ep == num_episodes - 1),
        name_prefix="a2c"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env_info = base_env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    max_action = float(env_info['action_high'][0])  # Get max_action from environment

    agent = A2CAgent(state_dim, action_dim, max_action, device)  # Pass max_action to agent

    best_reward = float('-inf')

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        states, actions, rewards, dones = [], [], [], []

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Train the agent using the collected trajectory
        trajectories = (states, actions, rewards, dones, None)
        metrics = agent.train(trajectories)

        # Log metrics to WandB and print to console
        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "actor_loss": metrics["actor_loss"],
            "critic_loss": metrics["critic_loss"],
            "entropy": metrics["entropy"]
        })
        print(f"Episode {episode + 1}/{num_episodes}, "
              f"Reward: {episode_reward:.2f}, "
              f"Actor Loss: {metrics['actor_loss']:.4f}, "
              f"Critic Loss: {metrics['critic_loss']:.4f}, "
              f"Entropy: {metrics['entropy']:.4f}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_dir = "models"
            os.makedirs(os.path.join(model_dir, "a2c"), exist_ok=True)
            torch.save(agent.actor.state_dict(),
                        f"{model_dir}/a2c/best_actor.pth")
            torch.save(agent.critic.state_dict(),
                        f"{model_dir}/a2c/best_critic.pth")
            wandb.log({"best_reward": best_reward})

        # Save periodic checkpoints
        if (episode % episode_trigger_count == 0) or (episode == num_episodes - 1):
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)

            checkpoint_path_actor = f"{model_dir}/a2c/ep_{episode}_actor.pth"
            checkpoint_path_critic = f"{model_dir}/a2c/ep_{episode}_critic.pth"

            # Save checkpoints
            torch.save(agent.actor.state_dict(), checkpoint_path_actor)
            torch.save(agent.critic.state_dict(), checkpoint_path_critic)

            # Log checkpoints to WandB
            wandb.save(checkpoint_path_actor)
            wandb.save(checkpoint_path_critic)

            # Log videos to WandB
            wandb.log({
                "video": wandb.Video(
                    f"{video_dir}/a2c-episode-{episode}.mp4",
                    format="mp4"
                )
            })

    wandb.finish()
    env.close()
    return agent
