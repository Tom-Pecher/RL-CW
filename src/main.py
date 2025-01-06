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
            
        # case 'a2c':
        #    from agents.a2c.train import train_agent
        #    agent = train_agent(args.hardcore, args.render)

        # case 'a3c':
        #    from agents.a3c.train import train_agent

        case 'ddpg':
            from agents.ddpg.train import train_agent

        case 'dqn':
            from agents.dqn.train import train_agent

        case 'drq_v2':
            from agents.drq_v2.train import train_agent

        case 'ppo':
            from agents.ppo.train import train_agent

        case 'random':
            from agents.random.train import train_agent

        case 'sac':
            from agents.sac.train import train_agent

        case 'sunrise':
            from agents.sunrise.train import train_agent

        case 'td3':
            from agents.td3.train import train_agent

        case 'trpo':
            from agents.trpo.train import train_agent
            agent = train_agent(args.hardcore, args.render)

        case _:
            print(f"Invalid agent: {args.agent}", file=stderr)
            return

    agent = train_agent(args.hardcore, args.render)

if __name__ == '__main__':
    main()
