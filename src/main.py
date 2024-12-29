"""
This is the main entry point for the project.
Please do not modify this file other than to add your own agent. 
All training code should be in the respective agent directory.
"""


import argparse
from sys import stderr
import torch

def main():
    """
    The main entry point for the project.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='', help='The agent to train.')
    parser.add_argument('--hardcore', action='store_true', help='Whether to use the hardcore version of the environment.')
    parser.add_argument('--render', action='store_true', help='Whether to render the environment.')
    args = parser.parse_args()
    
    # get right device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')



    match args.agent:
        case '':
            parser.print_help()
            return

        case 'example':
            from example import train_agent

        case 'actor-critic':
            from agents.actor_critic.train import train_agent

        case 'ddpg':
            from agents.ddpg.train import train_agent

        case 'dqn':
            from agents.dqn.train import train_agent

        case 'drq_v2':
            from agents.drq_v2.train import train_agent

        case 'ppo':
            from agents.ppo.train import train_agent

        case 'sac':
            from agents.sac.train import train_agent

        case 'td3':
            from agents.td3.train import train_agent

        case _:
            print(f"Invalid agent: {args.agent}", file=stderr)
            return

    agent = train_agent(args.hardcore, args.render, device)

if __name__ == '__main__':
    main()
