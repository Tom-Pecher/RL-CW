"""
Record a video of the TRPO agent.
"""

import torch
from bipedal_walker.environment import BipedalWalkerEnv
from agents.trpo.agent import TRPOAgent
from gym.wrappers.record_video import RecordVideo

def record_agent(model_path: str, hardcore: bool = False) -> None:
    # Create environment
    base_env = BipedalWalkerEnv(hardcore=hardcore, render=False)

    # Setup video recording
    env = RecordVideo(
        base_env.env,
        video_folder="output",
        episode_trigger=lambda _: True,
        name_prefix="trpo"
    )

    # Setup device
    if torch.backends.mps.is_available():
        choice = input("Do you want to use MPS? (y/n):")
        if choice == "y":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Get environment info
    env_info = base_env.get_env_info()
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    max_action = float(env_info['action_high'][0])

    # Create agent and load model
    agent = TRPOAgent(state_dim, action_dim, max_action, device)
    agent.policy.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy.eval()

    # Run one episode
    state, _ = env.reset()
    episode_reward = 0

    with torch.no_grad(): 
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist = agent.policy.get_distribution(state_tensor)
            action = dist.mean.cpu().numpy()[0]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            if done:
                break

    print(f"Episode finished with reward: {episode_reward}")
    env.close()
