import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="绘制训练曲线")
    parser.add_argument("--metrics", default="results/metrics.jsonl")
    parser.add_argument("--out", default="results/reward_curve.png")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.metrics).read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes = [row["episode"] for row in rows if "episode" in row]
    rewards = [row["reward"] for row in rows if "reward" in row]
    moving_rewards = [row["moving_reward"] for row in rows if "moving_reward" in row]

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, rewards, alpha=0.35, label="reward")
    plt.plot(episodes, moving_rewards, label="moving_reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
