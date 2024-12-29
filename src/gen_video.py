"""
Records a video of the agent.
"""
# Taking input imports
import argparse
from sys import stderr
# environment imports 
from bipedal_walker.environment import BipedalWalkerEnv
from gym.wrappers.record_video import RecordVideo
import torch
def main():
    """
    The main entry point for the project.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='', help='The agent to train.')
    parser.add_argument('--hardcore', action='store_true', help='Whether to use the hardcore version of the environment.')
    parser.add_argument('--model', type=str, default='', help='The path to the model file.')
    parser.add_argument('--output_folder', type=str, default='output', help='The video output location.')
    args = parser.parse_args()

    base_env = BipedalWalkerEnv(hardcore=args.hardcore, render=False)
    env_info = base_env.get_env_info()
    # get right device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('cpu') # I don't know why but cpu is faster for my mac 
    else:
        device = torch.device('cpu')


    match args.agent:
        case '':
            parser.print_help()
            return
        case 'actor-critic':
            raise NotImplementedError("Implement on merge")
        case 'ddpg':
            from agents.ddpg.agent import DDPGAgent
            agent = DDPGAgent(env_info,device)
        case 'dqn':
            from agents.dqn.agent import DQNAgent
            agent = DQNAgent(env_info,device)
        case 'drq_v2':
            from agents.drq_v2.agent import DrQv2Agent 
            agent = DrQv2Agent(env_info,device)
        case 'ppo':
            raise NotImplementedError("Implement on merge")
        case 'sac':
            from agents.sac.agent import SACAgent
            agent = SACAgent(env_info,device)
        case 'td3':
            from agents.td3.agent import TD3Agent
            agent = TD3Agent(env_info,device) 
        case 'trpo':
            raise NotImplementedError("Implement on merge")
        case _:
            print(f"Invalid agent: {args.agent}", file=stderr)
            print("Valid agents are: [example, actor-critic, ddpg, dqn, ppo]", file=stderr)
            return

    # If there are models load them
    if args.model != '':
        agent.load_agent(args.model)

    # Setup video recording
    video_env = RecordVideo(
        base_env.env,
        video_folder=args.output_folder,
        episode_trigger=lambda _: True,
        name_prefix=agent.get_agent_name()
    )
    agent.record_agent(video_env)


if __name__ == '__main__':
    main()
