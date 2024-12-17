"""
Training script for the deep Q-learning agent.
"""

import torch
from bipedal_walker.environment import BipedalWalkerEnv

from agents.dqn.agent import DQNAgent

def train_agent(hardcore: bool, render: bool):
    """
    Runs an example agent with a single episode.
    """
    env = BipedalWalkerEnv(hardcore, render)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env_info = env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']

    agent = DQNAgent(state_dim, action_dim, device)

    num_episodes = 1000

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

        # Logging
        avg_loss = (episode_loss / steps) if steps > 0 else 0
        print(f"Log: Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")


    return agent
