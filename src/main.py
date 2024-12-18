"""
This is the main entry point for the project.
Please do not modify this file other than to add your own agent. 
All training code should be in the respective agent directory.
"""


import argparse
from sys import stderr


def main():
    """
    The main entry point for the project.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='', help='The agent to train.')
    parser.add_argument('--hardcore', action='store_true', help='Whether to use the hardcore version of the environment.')
    parser.add_argument('--render', action='store_true', help='Whether to render the environment.')
    args = parser.parse_args()

    match args.agent:
        case '':
            parser.print_help()
            return

        case 'example':
            from example import train_agent
            agent = train_agent(args.hardcore, args.render)

        case 'actor-critic':
            from agents.actor_critic.train import train_agent
            agent = train_agent(args.hardcore, args.render)

        case 'ddpg':
            from agents.ddpg.train import train_agent
            agent = train_agent(args.hardcore, args.render)

        case 'dqn':
            from agents.dqn.train import train_agent
            agent = train_agent(args.hardcore, args.render)

        case 'ppo':
            from agents.ppo.train import train_agent
            agent = train_agent(args.hardcore, args.render)

        case 'sac':
            from agents.sac.train import train_agent
            agent = train_agent(args.hardcore, args.render)

        case 'td3':
            from agents.td3.train import train_agent
            agent = train_agent(args.hardcore, args.render)

        case _:
            print(f"Invalid agent: {args.agent}", file=stderr)
            print("Valid agents are: [example, actor-critic, ddpg, dqn, ppo, sac, td3]", file=stderr)
            return

if __name__ == '__main__':
    main()
