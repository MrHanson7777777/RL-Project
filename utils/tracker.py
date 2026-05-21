import json
from pathlib import Path


class Tracker:
    def __init__(self, enabled=False, project="rl-lab", name=None, config=None, log_dir="results"):
        self.enabled = enabled
        self.wandb = None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = open(self.log_dir / "metrics.jsonl", "a", encoding="utf-8")

        if enabled:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(project=project, name=name, config=config or {})
            except ImportError as exc:
                raise RuntimeError("wandb 未安装，请先运行 `pip install wandb`，或去掉 `--wandb`。") from exc

    def log(self, metrics, step=None):
        row = dict(metrics)
        if step is not None:
            row["step"] = step
        self.jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.jsonl.flush()
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def finish(self):
        self.jsonl.close()
        if self.wandb is not None:
            self.wandb.finish()
