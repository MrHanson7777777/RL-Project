import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CMD = [
    sys.executable,
    "train.py",
    "--config",
    "config/ppo.yaml",
    "--env",
    "CartPole-v0",
    "--episodes",
    "5000",
    "--wandb-group",
    "ppo-seeds",
]


if __name__ == "__main__":
    for seed in range(5):
        subprocess.run(
            [*BASE_CMD, "--seed", str(seed), "--run-name", f"ppo-seed-{seed}"],
            cwd=ROOT,
            check=True,
        )
