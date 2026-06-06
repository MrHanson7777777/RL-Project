"""Evaluate one algorithm with multiple seeds.

The lab document asks for 5 seeds. For each seed, use the average reward of
the final 100 training episodes as the representative value, then report
Reward_max, Reward_average, and Reward_variance.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from algorithms import AGENTS
from train import load_config, make_env
from utils import seed_env, set_seed


def train_one_seed(algo, env_name, episodes, seed, last_n, agent_kwargs):
    set_seed(seed)
    env = make_env(env_name)
    seed_env(env, seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    kwargs = dict(agent_kwargs)
    kwargs["state_dim"] = state_dim
    kwargs["action_dim"] = action_dim
    agent = AGENTS[algo](**kwargs)

    rewards = []
    try:
        for ep in range(1, episodes + 1):
            metrics = agent.train(env)
            rewards.append(metrics["reward"])
            if ep == 1 or ep % 500 == 0:
                last_avg = float(np.mean(rewards[-last_n:]))
                print(
                    f"  seed={seed} ep={ep:05d} reward={metrics['reward']:.1f} "
                    f"last{last_n}_avg={last_avg:.1f}"
                )
    finally:
        env.close()

    return float(np.mean(rewards[-last_n:]))


def build_agent_kwargs(algo, cfg):
    agent_kwargs = {
        "hidden_dim": cfg.get("hidden_dim", 128),
        "gamma": cfg.get("gamma", 0.98),
        "lr": cfg.get("lr", 1e-3),
        "device": cfg.get("device", "cpu"),
    }
    if algo == "reinforce":
        agent_kwargs["entropy_coef"] = cfg.get("entropy_coef", 0.01)
    if algo == "ppo":
        agent_kwargs.update(
            {
                "lamda": cfg.get("lamda", 0.95),
                "epochs": cfg.get("epochs", 10),
                "eps": cfg.get("eps", 0.2),
            }
        )
    if algo in {"ddpg", "sac"}:
        for key in ("tau", "buffer_size", "batch_size", "minimal_size"):
            if cfg.get(key) is not None:
                agent_kwargs[key] = cfg[key]
    if algo == "sac" and cfg.get("target_entropy") is not None:
        agent_kwargs["target_entropy"] = cfg["target_entropy"]
    return agent_kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple seeds and compute reward max/average/variance."
    )
    parser.add_argument("--config", required=True, help="Path to an algorithm config yaml.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--last-n", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    algo = cfg.get("algo", "reinforce")
    env_name = cfg.get("env", "CartPole-v0")
    episodes = cfg.get("episodes", 5000)
    agent_kwargs = build_agent_kwargs(algo, cfg)

    print(f"\n{'=' * 55}")
    print(f"  algo: {algo}  env: {env_name}  episodes: {episodes}")
    print(f"  seeds: {args.seeds}  representative: final {args.last_n} episodes")
    print(f"{'=' * 55}\n")

    seed_rewards = []
    for seed in args.seeds:
        print(f">>> start seed={seed}")
        reward = train_one_seed(algo, env_name, episodes, seed, args.last_n, agent_kwargs)
        seed_rewards.append(reward)
        print(f"  seed={seed} last{args.last_n}_avg_reward={reward:.2f}\n")

    r_max = float(np.max(seed_rewards))
    r_mean = float(np.mean(seed_rewards))
    r_var = float(np.var(seed_rewards))

    print(f"\n{'=' * 55}")
    print(f"  {algo.upper()} - 5 seed summary")
    print(f"{'=' * 55}")
    print(f"  representative values: {[f'{r:.2f}' for r in seed_rewards]}")
    print(f"  Reward_max      = {r_max:.2f}")
    print(f"  Reward_average  = {r_mean:.2f}")
    print(f"  Reward_variance = {r_var:.2f}")
    print(f"{'=' * 55}\n")

    out_path = Path("results") / f"eval_{algo}_seeds.csv"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", f"last{args.last_n}_avg_reward"])
        for seed, reward in zip(args.seeds, seed_rewards):
            writer.writerow([seed, f"{reward:.4f}"])
        writer.writerow(["max", f"{r_max:.4f}"])
        writer.writerow(["average", f"{r_mean:.4f}"])
        writer.writerow(["variance", f"{r_var:.4f}"])

    print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
