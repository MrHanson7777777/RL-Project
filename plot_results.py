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


def run_label(row):
    if row.get("run_name"):
        return row["run_name"]
    algo = row.get("algo", "run")
    gamma = row.get("gamma")
    seed = row.get("seed")
    parts = [str(algo)]
    if gamma is not None:
        parts.append(f"gamma={gamma}")
    if seed is not None:
        parts.append(f"seed={seed}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Plot reward curves.")
    parser.add_argument("--metrics", nargs="+", default=["results/metrics.jsonl"])
    parser.add_argument("--out", default="results/reward_curve.png")
    parser.add_argument("--metric", default=None, help="Metric to plot, e.g. moving_reward or eval_reward.")
    parser.add_argument("--raw", action="store_true", help="Plot raw reward instead of moving reward.")
    args = parser.parse_args()

    rows = load_rows(args.metrics)
    groups = defaultdict(list)
    for row in rows:
        if "episode" in row and "reward" in row:
            groups[run_label(row)].append(row)

    plt.figure(figsize=(9, 5))
    metric_name = args.metric or ("reward" if args.raw else "moving_reward")
    for label, group_rows in groups.items():
        group_rows = sorted(group_rows, key=lambda row: row["episode"])
        points = [(row["episode"], row[metric_name]) for row in group_rows if metric_name in row]
        if not points:
            continue
        episodes = [episode for episode, _ in points]
        values = [value for _, value in points]
        plt.plot(episodes, values, label=label)

    plt.xlabel("episode")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
