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
    "ppo-ablation-clipping",
]

RUNS = [
    ("ppo-ablation-with-clip", ["--use-clip"]),
    ("ppo-ablation-no-clip", ["--no-use-clip"]),
]


if __name__ == "__main__":
    for name, extra in RUNS:
        subprocess.run([*BASE_CMD, "--run-name", name, *extra], cwd=ROOT, check=True)
