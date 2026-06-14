import argparse

import numpy as np

from algorithms import AGENTS
from utils import Tracker, reset_env, seed_env, set_seed, step_env


def load_config(path):
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 yaml 配置需要 PyYAML，请先运行 `pip install PyYAML`。") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _registry_has_env(gym, env_name):
    registry = gym.envs.registry
    if isinstance(registry, dict):
        return env_name in registry
    env_specs = getattr(registry, "env_specs", None)
    if env_specs is not None:
        return env_name in env_specs
    try:
        registry.spec(env_name)
        return True
    except Exception:
        return False


def _register_cartpole_v0_if_needed(gym):
    if _registry_has_env(gym, "CartPole-v0"):
        return

    try:
        from gymnasium.envs.classic_control.cartpole import CartPoleEnv
    except ImportError:
        from gym.envs.classic_control.cartpole import CartPoleEnv

    gym.register(
        id="CartPole-v0",
        entry_point=CartPoleEnv,
        max_episode_steps=200,
        reward_threshold=195.0,
    )


def make_env(env_name):
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    if env_name == "CartPole-v0":
        _register_cartpole_v0_if_needed(gym)
    return gym.make(env_name)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description="统一强化学习训练入口", parents=[config_parser])
    parser.add_argument("--algo", choices=AGENTS.keys(), default=config.get("algo", "reinforce"))
    parser.add_argument("--env", default=config.get("env", "CartPole-v0"))
    parser.add_argument("--episodes", type=int, default=config.get("episodes", 5000))
    parser.add_argument("--hidden-dim", type=int, default=config.get("hidden_dim", 128))
    parser.add_argument("--gamma", type=float, default=config.get("gamma", 0.98))
    parser.add_argument("--lr", type=float, default=config.get("lr", 1e-3))
    parser.add_argument("--actor-lr", type=float, default=config.get("actor_lr", None))
    parser.add_argument("--critic-lr", type=float, default=config.get("critic_lr", None))
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
    parser.add_argument("--tau", type=float, default=config.get("tau", None))
    parser.add_argument("--buffer-size", type=int, default=config.get("buffer_size", None))
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", None))
    parser.add_argument("--minimal-size", type=int, default=config.get("minimal_size", None))
    parser.add_argument("--target-entropy", type=float, default=config.get("target_entropy", None))
    parser.add_argument("--entropy-coef", type=float, default=config.get("entropy_coef", 0.01))
    parser.add_argument("--normalize-returns", type=str_to_bool, default=config.get("normalize_returns", True))
    parser.add_argument("--ppo-use-clip", type=str_to_bool, default=config.get("ppo_use_clip", True))
    parser.add_argument("--ddpg-use-target-critic", type=str_to_bool,
                        default=config.get("ddpg_use_target_critic", True))
    parser.add_argument("--sac-auto-alpha", type=str_to_bool, default=config.get("sac_auto_alpha", True))
    parser.add_argument("--alpha", type=float, default=config.get("alpha", 0.2))
    parser.add_argument("--eval-interval", type=int, default=config.get("eval_interval", 100))
    parser.add_argument("--eval-episodes", type=int, default=config.get("eval_episodes", 5))
    parser.add_argument("--wandb-mode", default=config.get("wandb_mode", "online"),
                        choices=["online", "offline", "disabled"])
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
    if args.algo == "reinforce":
        agent_kwargs["entropy_coef"] = args.entropy_coef
        agent_kwargs["normalize_returns"] = args.normalize_returns
    if args.algo == "actor_critic":
        if args.actor_lr is not None:
            agent_kwargs["actor_lr"] = args.actor_lr
        if args.critic_lr is not None:
            agent_kwargs["critic_lr"] = args.critic_lr
    if args.algo == "ppo":
        agent_kwargs.update(
            {
                "lamda": args.lamda,
                "epochs": args.epochs,
                "eps": args.eps,
                "ppo_use_clip": args.ppo_use_clip,
            }
        )
    if args.algo in {"ddpg", "sac"}:
        for key in ("tau", "buffer_size", "batch_size", "minimal_size"):
            value = getattr(args, key)
            if value is not None:
                agent_kwargs[key] = value
    if args.algo == "ddpg":
        agent_kwargs["ddpg_use_target_critic"] = args.ddpg_use_target_critic
    if args.algo == "sac" and args.target_entropy is not None:
        agent_kwargs["target_entropy"] = args.target_entropy
    if args.algo == "sac":
        agent_kwargs["sac_auto_alpha"] = args.sac_auto_alpha
        agent_kwargs["alpha"] = args.alpha
    return agent_cls(**agent_kwargs)


def select_eval_action(agent, state):
    import torch

    network = getattr(agent, "policy", None) or getattr(agent, "actor", None)
    if network is None:
        return agent.take_action(state)

    device = getattr(agent, "device", torch.device("cpu"))
    state_t = torch.tensor(state, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = network(state_t)
    return int(torch.argmax(probs).item())


def evaluate_agent(agent, env_name, seed, episodes):
    env = make_env(env_name)
    seed_env(env, seed)
    rewards = []
    try:
        for ep in range(episodes):
            state = reset_env(env, seed + ep)
            done = False
            total_reward = 0.0
            while not done:
                action = select_eval_action(agent, state)
                state, reward, done, _ = step_env(env, action)
                total_reward += reward
            rewards.append(total_reward)
    finally:
        env.close()
    return float(np.mean(rewards))


def main():
    args = parse_args()
    set_seed(args.seed)
    env = make_env(args.env)
    seed_env(env, args.seed)
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
        mode=args.wandb_mode,
    )

    recent_rewards = []
    try:
        for ep in range(1, args.episodes + 1):
            metrics = agent.train(env)
            recent_rewards.append(metrics["reward"])
            moving_reward = float(np.mean(recent_rewards[-100:]))
            log_metrics = {
                "algo": args.algo,
                "env": args.env,
                "gamma": args.gamma,
                "seed": args.seed,
                "run_name": args.run_name or f"{args.algo}-{args.env}",
                "episode": ep,
                "moving_reward": moving_reward,
                **metrics,
            }
            if args.algo == "actor_critic":
                log_metrics["actor_lr"] = args.actor_lr if args.actor_lr is not None else args.lr
                log_metrics["critic_lr"] = args.critic_lr if args.critic_lr is not None else args.lr
            if args.eval_interval > 0 and (ep == 1 or ep % args.eval_interval == 0):
                log_metrics["eval_reward"] = evaluate_agent(
                    agent,
                    args.env,
                    args.seed + ep * 1000,
                    args.eval_episodes,
                )
            tracker.log(log_metrics, step=ep)

            if ep == 1 or ep % 10 == 0:
                message = f"ep={ep:04d} reward={metrics['reward']:.1f} moving_reward={moving_reward:.1f}"
                if "eval_reward" in log_metrics:
                    message += f" eval_reward={log_metrics['eval_reward']:.1f}"
                print(message)
    finally:
        tracker.finish()
        env.close()


if __name__ == "__main__":
    main()
