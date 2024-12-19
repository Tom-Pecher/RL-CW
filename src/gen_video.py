"""
Records a video of the agent.
"""

import argparse
from sys import stderr

from agents.ddpg.agent import DDPGAgent
from agents.dqn.agent import DQNAgent

from bipedal_walker.environment import BipedalWalkerEnv
from gym.wrappers.record_video import RecordVideo

def main():
    """
    The main entry point for the project.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='', help='The agent to train.')
    parser.add_argument('--hardcore', action='store_true', help='Whether to use the hardcore version of the environment.')
    parser.add_argument('--model', type=str, default='', help='The path to the model file.')
    parser.add_argument('--output-folder', type=str, default='output', help='The video output location.')
    args = parser.parse_args()

    base_env = BipedalWalkerEnv(hardcore=args.hardcore, render=False)
    env_info = base_env.get_env_info()

    match args.agent:
        case '':
            parser.print_help()
            return
        case 'actor-critic':
            from agents.actor_critic.agent import A2CAgent
            agent = A2CAgent
        case 'ddpg':
            from agents.ddpg.agent import DDPGAgent
            agent = DDPGAgent
        case 'dqn':
            from agents.dqn.agent import DQNAgent
            agent = DQNAgent
        case 'ppo':
            from agents.ppo.agent import PPOAgent
            agent = PPOAgent
        case _:
            print(f"Invalid agent: {args.agent}", file=stderr)
            print("Valid agents are: [example, actor-critic, ddpg, dqn, ppo]", file=stderr)
            return


        # Setup video recording
    video_env = RecordVideo(
        base_env.env,
        video_folder=args.video_folder,
        episode_trigger=lambda _: True,
        name_prefix=agent.get_agent_name()
    )
    agent.record_agent(video_env)


if __name__ == '__main__':
    main()
