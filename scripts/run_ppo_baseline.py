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
]


def run(name, group="ppo-baseline", extra=None):
    cmd = [*BASE_CMD, "--wandb-group", group, "--run-name", name]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    run("ppo-baseline-cartpole-v0")
