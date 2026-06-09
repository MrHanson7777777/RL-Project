# RL Project

本项目用于完成“策略梯度算法对比实验”。实验文档要求：

- 环境：`CartPole-v0`
- 训练轮次：`episodes=5000`
- 折扣因子对比：必须跑 `gamma=0.95` 和 `gamma=0.5`
- 每个算法选择 5 个随机种子：`seed=0,1,2,3,4`
- 测试指标：取训练最后 100 轮 reward 平均值，统计最大值、平均值、方差
- 选择 `DDPG / PPO / SAC` 时，还需要额外做一个消融实验

## 环境准备

```bash
conda activate rl_env
```

如果要直接在线上传到 W&B，先确保当前 PowerShell 能走代理：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
$env:ALL_PROXY="http://127.0.0.1:7890"
```

下面所有 `train.py` 命令都显式加了 `--wandb-mode online`，运行结束后会直接上传到 W&B。若网络不稳定，可把 `online` 改成 `offline`，训练结束后再执行：

```powershell
python -m wandb sync wandb\offline-run-xxxx
```

## Actor-Critic

Actor-Critic 是必做算法，也需要跑两个 gamma，每个 gamma 跑 5 个 seed。

另外，按照双时间尺度的验证思路，还要做一组 actor / critic 学习率比例实验。这里固定 critic 学习率为 `0.003`，actor 学习率分别取 `0.0003`、`0.003`、`0.00003`，对应比例 `0.1`、`1.0`、`0.01`。

### Actor-Critic, gamma=0.95

```powershell
python train.py --config config/actor_critic.yaml --gamma 0.95 --seed 0 --run-name actor-critic-gamma0.95-seed0 --wandb-mode online
python train.py --config config/actor_critic.yaml --gamma 0.95 --seed 1 --run-name actor-critic-gamma0.95-seed1 --wandb-mode online
python train.py --config config/actor_critic.yaml --gamma 0.95 --seed 2 --run-name actor-critic-gamma0.95-seed2 --wandb-mode online
python train.py --config config/actor_critic.yaml --gamma 0.95 --seed 3 --run-name actor-critic-gamma0.95-seed3 --wandb-mode online
python train.py --config config/actor_critic.yaml --gamma 0.95 --seed 4 --run-name actor-critic-gamma0.95-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/actor_critic.yaml --gamma 0.95
```

### Actor-Critic, gamma=0.5

```powershell
python train.py --config config/actor_critic.yaml --gamma 0.5 --seed 0 --run-name actor-critic-gamma0.5-seed0 --wandb-mode online
python train.py --config config/actor_critic.yaml --gamma 0.5 --seed 1 --run-name actor-critic-gamma0.5-seed1 --wandb-mode online
python train.py --config config/actor_critic.yaml --gamma 0.5 --seed 2 --run-name actor-critic-gamma0.5-seed2 --wandb-mode online
python train.py --config config/actor_critic.yaml --gamma 0.5 --seed 3 --run-name actor-critic-gamma0.5-seed3 --wandb-mode online
python train.py --config config/actor_critic.yaml --gamma 0.5 --seed 4 --run-name actor-critic-gamma0.5-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/actor_critic.yaml --gamma 0.5
```

### Actor-Critic 学习率比例实验, gamma=0.95

#### 0.1 比例：actor_lr=0.0003, critic_lr=0.003

```powershell
python train.py --config config/actor_critic_lr_actor0p0003_critic0p003.yaml --seed 0 --run-name actor-critic-lr0.0003-vs-0.003-gamma0.95-seed0 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p0003_critic0p003.yaml --seed 1 --run-name actor-critic-lr0.0003-vs-0.003-gamma0.95-seed1 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p0003_critic0p003.yaml --seed 2 --run-name actor-critic-lr0.0003-vs-0.003-gamma0.95-seed2 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p0003_critic0p003.yaml --seed 3 --run-name actor-critic-lr0.0003-vs-0.003-gamma0.95-seed3 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p0003_critic0p003.yaml --seed 4 --run-name actor-critic-lr0.0003-vs-0.003-gamma0.95-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/actor_critic_lr_actor0p0003_critic0p003.yaml
```

#### 1.0 比例：actor_lr=0.003, critic_lr=0.003

```powershell
python train.py --config config/actor_critic_lr_actor0p003_critic0p003.yaml --seed 0 --run-name actor-critic-lr0.003-vs-0.003-gamma0.95-seed0 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p003_critic0p003.yaml --seed 1 --run-name actor-critic-lr0.003-vs-0.003-gamma0.95-seed1 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p003_critic0p003.yaml --seed 2 --run-name actor-critic-lr0.003-vs-0.003-gamma0.95-seed2 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p003_critic0p003.yaml --seed 3 --run-name actor-critic-lr0.003-vs-0.003-gamma0.95-seed3 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p003_critic0p003.yaml --seed 4 --run-name actor-critic-lr0.003-vs-0.003-gamma0.95-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/actor_critic_lr_actor0p003_critic0p003.yaml
```

#### 0.01 比例：actor_lr=0.00003, critic_lr=0.003

```powershell
python train.py --config config/actor_critic_lr_actor0p00003_critic0p003.yaml --seed 0 --run-name actor-critic-lr0.00003-vs-0.003-gamma0.95-seed0 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p00003_critic0p003.yaml --seed 1 --run-name actor-critic-lr0.00003-vs-0.003-gamma0.95-seed1 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p00003_critic0p003.yaml --seed 2 --run-name actor-critic-lr0.00003-vs-0.003-gamma0.95-seed2 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p00003_critic0p003.yaml --seed 3 --run-name actor-critic-lr0.00003-vs-0.003-gamma0.95-seed3 --wandb-mode online
python train.py --config config/actor_critic_lr_actor0p00003_critic0p003.yaml --seed 4 --run-name actor-critic-lr0.00003-vs-0.003-gamma0.95-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/actor_critic_lr_actor0p00003_critic0p003.yaml
```


## 画图

训练曲线：

```powershell
python plot_results.py --metrics results/metrics.jsonl --out results/reward_curve.png
```

测试曲线：

```powershell
python plot_results.py --metrics results/metrics.jsonl --metric eval_reward --out results/eval_curve.png
```

原始每轮 reward 曲线：

```powershell
python plot_results.py --metrics results/metrics.jsonl --raw --out results/raw_reward_curve.png
```

## 结果文件

- `results/metrics.jsonl`：所有训练日志
- `results/eval_*_seeds.csv`：`eval_seeds.py` 生成的 5-seed 统计表
- `results/reward_curve.png`：训练 reward 曲线
- `results/eval_curve.png`：测试 reward 曲线
