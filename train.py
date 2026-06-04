import argparse

import numpy as np

from algorithms import AGENTS
from utils import Tracker, set_seed


def load_config(path):
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 yaml 配置需要 PyYAML，请先运行 `pip install PyYAML`。") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_env(env_name):
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    return gym.make(env_name)


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description="统一强化学习训练入口", parents=[config_parser])
    parser.add_argument("--algo", choices=AGENTS.keys(), default=config.get("algo", "reinforce"))
    parser.add_argument("--env", default=config.get("env", "CartPole-v1"))
    parser.add_argument("--episodes", type=int, default=config.get("episodes", 300))
    parser.add_argument("--hidden-dim", type=int, default=config.get("hidden_dim", 128))
    parser.add_argument("--gamma", type=float, default=config.get("gamma", 0.98))
    parser.add_argument("--lr", type=float, default=config.get("lr", 1e-3))
    parser.add_argument("--seed", type=int, default=config.get("seed", 0))
    parser.add_argument("--device", default=config.get("device", "cpu"))
    parser.add_argument("--wandb", action="store_true", default=config.get("wandb", False))
    parser.add_argument("--wandb-entity", default=config.get("wandb_entity", None))
    parser.add_argument("--wandb-project", default=config.get("wandb_project", "rl-lab"))
    parser.add_argument("--wandb-group", default=config.get("wandb_group", None))
    parser.add_argument("--run-name", default=config.get("run_name", None))
    parser.add_argument("--lamda", type=float, default=config.get("lamda", 0.95))
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 10))
    parser.add_argument("--eps", type=float, default=config.get("eps", 0.2))
    parser.add_argument("--use-clip", action=argparse.BooleanOptionalAction, default=config.get("use_clip", True))
    parser.add_argument("--tau", type=float, default=config.get("tau", None))
    parser.add_argument("--buffer-size", type=int, default=config.get("buffer_size", None))
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", None))
    parser.add_argument("--minimal-size", type=int, default=config.get("minimal_size", None))
    parser.add_argument("--target-entropy", type=float, default=config.get("target_entropy", None))
    return parser.parse_args()


def build_agent(args, state_dim, action_dim):
    agent_cls = AGENTS[args.algo]
    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "gamma": args.gamma,
        "lr": args.lr,
        "device": args.device,
    }
    if args.algo == "ppo":
        agent_kwargs.update(
            {
                "lamda": args.lamda,
                "epochs": args.epochs,
                "eps": args.eps,
                "use_clip": args.use_clip,
            }
        )
    if args.algo in {"ddpg", "sac"}:
        for key in ("tau", "buffer_size", "batch_size", "minimal_size"):
            value = getattr(args, key)
            if value is not None:
                agent_kwargs[key] = value
    if args.algo == "sac" and args.target_entropy is not None:
        agent_kwargs["target_entropy"] = args.target_entropy
    return agent_cls(**agent_kwargs)


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
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.run_name or f"{args.algo}-{args.env}",
        group=args.wandb_group,
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
