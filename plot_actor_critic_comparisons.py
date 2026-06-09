import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def actor_critic_rows(rows):
    for row in rows:
        if row.get("algo") == "actor_critic" and "episode" in row and "moving_reward" in row:
            yield row


def select_runs(rows, predicate):
    grouped = defaultdict(list)
    for row in actor_critic_rows(rows):
        if predicate(row):
            grouped[row.get("run_name", f"seed={row.get('seed')}")].append(row)
    return grouped


def average_series(run_groups):
    if not run_groups:
        return [], []

    episode_values = defaultdict(list)
    for rows in run_groups.values():
        for row in rows:
            episode_values[row["episode"]].append(row["moving_reward"])

    episodes = sorted(episode_values)
    values = [sum(episode_values[episode]) / len(episode_values[episode]) for episode in episodes]
    return episodes, values


def plot_grouped_runs(grouped_runs, out_path, title, xlabel="episode", ylabel="moving_reward"):
    plt.figure(figsize=(10, 5.5))
    for label, rows in sorted(grouped_runs.items()):
        rows = sorted(rows, key=lambda item: item["episode"])
        episodes = [row["episode"] for row in rows]
        rewards = [row["moving_reward"] for row in rows]
        plt.plot(episodes, rewards, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_average_curves(series_map, out_path, title):
    plt.figure(figsize=(10, 5.5))
    for label, (episodes, values) in series_map.items():
        plt.plot(episodes, values, label=label)

    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel("moving_reward")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Actor-Critic comparison curves.")
    parser.add_argument("--metrics", nargs="+", default=["results/metrics.jsonl"])
    parser.add_argument("--out-dir", default="results/plots")
    args = parser.parse_args()

    rows = load_rows(args.metrics)
    out_dir = Path(args.out_dir)

    gamma_average_series = {
        "gamma=0.95": average_series(
            select_runs(rows, lambda row: row.get("run_name", "").startswith("actor-critic-gamma0.95-seed"))
        ),
        "gamma=0.5": average_series(
            select_runs(rows, lambda row: row.get("run_name", "").startswith("actor-critic-gamma0.5-seed"))
        ),
    }
    plot_average_curves(
        gamma_average_series,
        out_dir / "actor_critic_gamma_moving_reward.png",
        "Actor-Critic moving_reward comparison by gamma",
    )

    lr_average_series = {
        "actor_lr=0.0003": average_series(
            select_runs(
                rows,
                lambda row: row.get("run_name", "").startswith(
                    "actor-critic-lr0.0003-vs-0.003-gamma0.95-seed"
                ),
            )
        ),
        "actor_lr=0.003": average_series(
            select_runs(
                rows,
                lambda row: row.get("run_name", "").startswith(
                    "actor-critic-lr0.003-vs-0.003-gamma0.95-seed"
                ),
            )
        ),
        "actor_lr=0.00003": average_series(
            select_runs(
                rows,
                lambda row: row.get("run_name", "").startswith(
                    "actor-critic-lr0.00003-vs-0.003-gamma0.95-seed"
                ),
            )
        ),
    }
    plot_average_curves(
        lr_average_series,
        out_dir / "actor_critic_lr_ratio_moving_reward.png",
        "Actor-Critic moving_reward comparison by actor/critic lr ratio",
    )

    seed_compare = select_runs(
        rows,
        lambda row: row.get("run_name", "").startswith("actor-critic-gamma0.95-seed"),
    )
    plot_grouped_runs(
        seed_compare,
        out_dir / "actor_critic_seed_moving_reward.png",
        "Actor-Critic moving_reward comparison by seed",
    )

    print("saved:")
    print(out_dir / "actor_critic_gamma_moving_reward.png")
    print(out_dir / "actor_critic_lr_ratio_moving_reward.png")
    print(out_dir / "actor_critic_seed_moving_reward.png")


if __name__ == "__main__":
    main()