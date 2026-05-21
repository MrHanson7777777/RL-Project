import argparse

import numpy as np

from algorithms import AGENTS
from utils import Tracker, set_seed


def make_env(env_name):
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    return gym.make(env_name)


def parse_args():
    parser = argparse.ArgumentParser(description="统一强化学习训练入口")
    parser.add_argument("--algo", choices=AGENTS.keys(), default="reinforce")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="rl-lab")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_agent(args, state_dim, action_dim):
    agent_cls = AGENTS[args.algo]
    return agent_cls(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        device=args.device,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    env = make_env(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = build_agent(args, state_dim, action_dim)

    config = vars(args)
    tracker = Tracker(
        enabled=args.wandb,
        project=args.wandb_project,
        name=args.run_name or f"{args.algo}-{args.env}",
        config=config,
    )

    recent_rewards = []
    try:
        for ep in range(1, args.episodes + 1):
            metrics = agent.train(env)
            recent_rewards.append(metrics["reward"])
            moving_reward = float(np.mean(recent_rewards[-20:]))
            log_metrics = {"episode": ep, "moving_reward": moving_reward, **metrics}
            tracker.log(log_metrics, step=ep)

            if ep == 1 or ep % 10 == 0:
                print(f"ep={ep:04d} reward={metrics['reward']:.1f} moving_reward={moving_reward:.1f}")
    finally:
        tracker.finish()
        env.close()


if __name__ == "__main__":
    main()
