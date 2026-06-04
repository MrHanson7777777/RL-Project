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
    "ppo-gamma",
]


def run(gamma):
    subprocess.run(
        [*BASE_CMD, "--gamma", str(gamma), "--run-name", f"ppo-gamma-{str(gamma).replace('.', '')}"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    for gamma in (0.95, 0.5):
        run(gamma)
