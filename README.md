# RL Project

强化学习课程第二组代码管理。当前仓库按课程实验常用结构组织，多个强化学习算法共用 `networks/mlp.py` 中的 MLP 网络组件，并提供统一训练入口和可选 wandb 监控。

## 目录结构

```text
RL-Project/
├── README.md
├── requirements.txt
├── train.py                 # 统一训练入口，支持 wandb
├── plot_results.py          # 绘制 results/metrics.jsonl 曲线
├── run_sweep.py             # 本地批量运行多个算法
├── algorithms/
│   ├── reinforce.py
│   ├── actor_critic.py
│   ├── ppo.py
│   ├── ddpg.py
│   └── sac.py
├── networks/
│   └── mlp.py               # PolicyNetwork / ValueNetwork / QNetwork 共用 MLP
├── utils/
│   ├── experiment.py        # 随机种子、Gym API 兼容
│   ├── replay_buffer.py
│   └── tracker.py           # JSONL + wandb 训练日志
└── results/
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练

默认不开 wandb，只写入 `results/metrics.jsonl`。实验文档要求使用 `CartPole-v0`，训练轮次 `episodes=5000`：

```bash
python train.py --algo reinforce --env CartPole-v0 --episodes 5000 --gamma 0.95
```

支持的算法：

```text
reinforce, actor_critic, ppo, sac, ddpg
```

开启 wandb：

```bash
wandb login
python train.py --config config/ppo.yaml
```

PPO 分工实验脚本位于 `scripts/`，包括 baseline、gamma 对比、clip range 调参、clipping 消融和 5 个随机种子实验。实验默认同步到 wandb 的 `23281269- / RL Project`。

## PPO 实验脚本

```bash
python scripts/run_ppo_baseline.py
python scripts/run_ppo_gamma.py
python scripts/run_ppo_tuning.py
python scripts/run_ppo_ablation.py
python scripts/run_ppo_seeds.py
```

其中 `run_ppo_gamma.py` 会按要求运行 `gamma=0.95` 和 `gamma=0.5`；`run_ppo_tuning.py` 用于 PPO 原论文核心参数 clip range，即 `eps=0.1/0.2/0.3` 的对比。

## 绘图

```bash
python plot_results.py --metrics results/metrics.jsonl --out results/reward_curve.png
```
