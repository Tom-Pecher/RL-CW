"""
Records a video of the agent.
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
    parser.add_argument('--model', type=str, default='', help='The path to the model file.')
    args = parser.parse_args()

    match args.agent:
        case '':
            parser.print_help()
            return

        case 'actor-critic':
            from agents.actor_critic.record import record_agent
            record_agent(args.model, args.hardcore)

        case 'ddpg':
            from agents.ddpg.record import record_agent
            record_agent(args.model, args.hardcore)

        case 'dqn':
            from agents.dqn.record import record_agent
            record_agent(args.model, args.hardcore)

        case 'ppo':
            from agents.ppo.record import record_agent
            record_agent(args.model, args.hardcore)

        case 'sac':
            from agents.sac.record import record_agent
            record_agent(args.model, args.hardcore)

        case 'td3':
            from agents.td3.record import record_agent
            record_agent(args.model, args.hardcore)

        case _:
            print(f"Invalid agent: {args.agent}", file=stderr)
            print("Valid agents are: [actor-critic, ddpg, dqn, ppo, sac, td3]", file=stderr)
            return

if __name__ == '__main__':
    main()
