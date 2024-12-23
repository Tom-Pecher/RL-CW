"""
Training script for the TD3 agent.
"""

import torch
import os
import wandb

from bipedal_walker.environment import BipedalWalkerEnv
from agents.trpo.agent import TRPOAgent

from gym.wrappers.record_video import RecordVideo


def train_agent(hardcore: bool, render: bool):
    """
    Trains the TRPO agent.
    """

    num_episodes = 3000

    # Initialize WandB
    wandb.init(
        project="bipedal-walker",
        config={
            "algorithm": "TRPO",
            "environment": "BipedalWalker-v3",
            "hardcore": hardcore,
            "num_episodes": num_episodes,
            "gamma": 0.99,
            "max_kl": 0.005,
            "damping": 0.1,
            "entropy_coef": 0.01,
            "value_lr": 1e-3,
            "max_step_size": 0.01,
            "backtrack_coeff": 0.5,
            "l2_reg": 0.01,
            "network_size": [64, 64]
        }
    )

    # Create environment
    base_env = BipedalWalkerEnv(hardcore, render)

    # Create output dir for videos
    video_dir = "videos/trpo"
    os.makedirs(video_dir, exist_ok=True)

    # Setup video recording
    env = RecordVideo(
        base_env.env,
        video_dir,
        episode_trigger=lambda ep: ep % 1000 == 0,
        name_prefix="video"
    )

    if torch.backends.mps.is_available():
        choice = input("Do you want to use MPS? (Y/N):")
        if choice == "Y":
            device = torch.device("mps") # added mps for apple people
        else:
            device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    env_info = base_env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    max_action = float(env_info['action_high'][0])

    agent = TRPOAgent(state_dim, action_dim, max_action, device)
    best_reward = float('-inf')
    running_reward = 0

    agent.policy.load_state_dict(torch.load("models/trpo/trpo_best_policy.pth"))

    for episode in range(num_episodes):

        # COLLECT DATA 
        trajectories = agent.collect_data(env, min_timesteps=2048)
        advantages, returns = agent.estimate_advantage(trajectories[0])

        states = torch.FloatTensor(trajectories[0]['states']).to(device)
        actions = torch.FloatTensor(trajectories[0]['actions']).to(device)
        old_probabilitys = torch.FloatTensor(trajectories[0]['probabilitys']).to(device)

        # COMPUTE SEARCH DIRECTION 
        step_dir = agent.compute_search_direction(states, actions, advantages, old_probabilitys)

        # LINE SEARCH 
        step_size = agent.line_search(states, actions, advantages, old_probabilitys, step_dir)

        policy_updated = False
        kl = 0.0
        improvement = 0.0

        if step_size > 0:
            
            old_params = []
            for param in agent.policy.parameters():
                old_params.append(param.data.view(-1))
            old_params = torch.cat(old_params)
            
            new_params = old_params + step_size * step_dir
            # COMPUTE SURROGATE LOSS 
            old_loss = agent.compute_surrogate_loss(states, actions, advantages, old_probabilitys)
            # GET DISTRIBUTION 
            old_dist = agent.policy.get_distribution(states)
            # UPDATE POLICY PARAMS 
            agent.update_policy_params(new_params)
            policy_updated = True
            # COMPUTE KL DIVERGENCE 
            kl = agent.compute_kl_constraint(states, old_dist)
            # COMPUTE NEW LOSS 
            new_loss = agent.compute_surrogate_loss(states, actions, advantages, old_probabilitys)
            # COMPUTE IMPROVEMENT 
            improvement = (old_loss - new_loss).item() if isinstance(new_loss, torch.Tensor) else old_loss - new_loss

        # VALUE UPDATE 
        value_loss = agent.update_value(trajectories[0]['states'], returns)

        # LOGGING 
        episode_reward = sum(trajectories[0]['rewards'])
        if running_reward == 0: running_reward = episode_reward
        else:running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # print to the terminal 
        print(f"Episode {episode + 1} | "
              f"Reward: {episode_reward:.7f} | "
              f"Running reward: {running_reward:.7f} | "
              f"KL: {kl:.7f} | "
              f"Step size: {step_size:.7f} | "
              f"Policy updated: {policy_updated}"
            )

        # Log metrics to WandB and print to console
        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "running_reward": running_reward,
            "value_loss": value_loss,
            "kl_divergence": kl,
            "policy_step_size": step_size,
            "policy_improvement": improvement,
            "trajectory_length": len(trajectories[0]['states']),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "steps": len(trajectories[0]['states']),
            "policy_updated": policy_updated
        })

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            print(f"New best reward: {best_reward:.4f}")
            os.makedirs("models", exist_ok=True)
            torch.save(agent.policy.state_dict(), "models/trpo/trpo_best_policy.pth")
            torch.save(agent.value.state_dict(), "models/trpo/trpo_best_value.pth")

        
    # wandb.finish()
    env.close()
    return agent 
