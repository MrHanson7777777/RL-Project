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

默认不开 wandb，只写入 `results/metrics.jsonl`：

```bash
python train.py --algo reinforce --env CartPole-v1 --episodes 300
```

支持的算法：

```text
reinforce, actor_critic, ppo, sac, ddpg
```

开启 wandb：

```bash
wandb login
python train.py --algo ppo --env CartPole-v1 --episodes 300 --wandb --wandb-project rl-lab
```

## 绘图

```bash
python plot_results.py --metrics results/metrics.jsonl --out results/reward_curve.png
```
