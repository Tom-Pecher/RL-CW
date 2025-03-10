
"""
Training script for the PPO agent.
"""

from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import os
import wandb
import numpy as np

from agents.ppo.agent import PPOAgent
from agents.common.replay_buffer import ReplayBuffer

from bipedal_walker.environment import BipedalWalkerEnv
from gym.wrappers.record_video import RecordVideo

from typing import Tuple

def run_policy(env: RecordVideo, agent: PPOAgent, T: int) -> tuple[list,list,list,list,list,bool]:

    tau = [[],[],[],[],[], False]
    state, _ = env.reset()
    for step in range(T):
        action,log_prob = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition in tau
        for index, item in enumerate((state,action,log_prob,reward,next_state)):
            tau[index].append(item)
        tau[-1] = terminated

        state = next_state
        if done:
            break
    
    return tuple(tau)


def collect_trajectories(env:RecordVideo,agent: PPOAgent, total_timesteps) -> Tuple[ReplayBuffer, float]:
    """
    Generates a ReplayBuffer of different trajectories of the same policy  
    Args:
        env (gym.Env): The environment which the agent acts in.
        agent (PPOAgent): The agent within the environment.
        num_traj: the number of trajectories that should be generated
    Returns:
        ReplayBuffer: A replay buffer name up of the trajectories
        float: Average reward per trajectory
    """
    buffer = ReplayBuffer(100000)
    total_reward = 0
    num_traj = 0
    current_timesteps = 0
    agent.training = True
    while current_timesteps < total_timesteps:
        # Definite possibility of multithreading
        states, actions, log_probs, rewards, next_states, terminates = run_policy(env,agent, total_timesteps - current_timesteps)

        advantages, rewards_to_go = agent.compute_gae(states,rewards,next_states,terminates)
        for state, action,log_prob, advantage, reward_to_go in zip(states,actions,log_probs, advantages, rewards_to_go):
            buffer.push(state,action,log_prob, advantage,reward_to_go)  

        total_reward += sum(rewards) 
        num_traj += 1
        current_timesteps += len(rewards)

    return buffer, total_reward / num_traj

def train_agent(hardcore: bool, render: bool):
    """
    Runs an example agent with a single episode.
    """
    base_env = BipedalWalkerEnv(hardcore, render)

    # Create output dir for videos
    video_dir = "videos/ppo"
    os.makedirs(video_dir, exist_ok=True)

    iteration_trigger_count = 100

    # Setup video recording
    env = RecordVideo(
        base_env.env,
        video_dir,
        episode_trigger = lambda ep: ep % iteration_trigger_count == 0,
        name_prefix="video"
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env_info = base_env.get_env_info()
    agent = PPOAgent(env_info,device)

    num_iterations = 10000
    best_reward = float('-inf')
    for iteration in range(num_iterations):

        # Collects trajectories
        buffer, avg_reward = collect_trajectories(env,agent,2048)

        # Updates the best reward if the avg traj reward is better 

        loss = 0
        # Train policy network 
        for _ in range(agent.num_epoch):
            batch = buffer.sample(agent.batch_size,device) 
            states,actions, log_probs, advantages,rewards_to_go  = batch
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss += agent.train(states,actions,log_probs,advantages,rewards_to_go)

        avg_loss = loss / (len(buffer) * agent.num_epoch)

        print(f"Log: iteration {iteration + 1}, "
              f"Reward: {avg_reward:.2f}, "
              f"Avg loss: {avg_loss:.4f}, "
              f"Epsilon: {agent.epsilon:.4f}")

        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            model_dir = "models"
            os.makedirs(os.path.join(model_dir, "ppo"), exist_ok=True)
            torch.save(agent.policy_net.state_dict(),
                        f"{model_dir}/ppo/best_policy.pth")


        # Save periodic checkpoints
        if (iteration) % iteration_trigger_count == 0:
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)

            checkpoint_path_policy = f"{model_dir}/ppo/ep_{iteration}_policy_net.pth"
            checkpoint_path_value = f"{model_dir}/ppo/ep_{iteration}_value_net.pth"

            # Save checkpoints
            torch.save(agent.policy_net.state_dict(), checkpoint_path_policy)
            torch.save(agent.value_net.state_dict(),checkpoint_path_value)

    env.close()
    return agent
