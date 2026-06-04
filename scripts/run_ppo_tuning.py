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

EXPERIMENTS = [
    ("eps", "ppo-tuning-eps", "--eps", [0.1, 0.2, 0.3]),
    ("lambda", "ppo-tuning-lambda", "--lamda", [0.90, 0.95, 0.99]),
    ("epochs", "ppo-tuning-epochs", "--epochs", [3, 10, 20]),
    ("lr", "ppo-tuning-lr", "--lr", [0.0003, 0.001, 0.003]),
]


def fmt(value):
    return str(value).replace(".", "")


def run(label, group, flag, value):
    subprocess.run(
        [*BASE_CMD, "--wandb-group", group, flag, str(value), "--run-name", f"ppo-{label}-{fmt(value)}"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    for label, group, flag, values in EXPERIMENTS:
        for value in values:
            run(label, group, flag, value)
