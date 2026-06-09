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

## REINFORCE

REINFORCE 需要跑两个 gamma，每个 gamma 跑 5 个 seed。

### REINFORCE, gamma=0.95

```powershell
python train.py --config config/reinforce_gamma095.yaml --seed 0 --run-name reinforce-gamma0.95-seed0 --wandb-mode online
python train.py --config config/reinforce_gamma095.yaml --seed 1 --run-name reinforce-gamma0.95-seed1 --wandb-mode online
python train.py --config config/reinforce_gamma095.yaml --seed 2 --run-name reinforce-gamma0.95-seed2 --wandb-mode online
python train.py --config config/reinforce_gamma095.yaml --seed 3 --run-name reinforce-gamma0.95-seed3 --wandb-mode online
python train.py --config config/reinforce_gamma095.yaml --seed 4 --run-name reinforce-gamma0.95-seed4 --wandb-mode online
```

生成 5-seed 统计表：

```powershell
python eval_seeds.py --config config/reinforce_gamma095.yaml
```

### REINFORCE, gamma=0.5

```powershell
python train.py --config config/reinforce_gamma05.yaml --seed 0 --run-name reinforce-gamma0.5-seed0 --wandb-mode online
python train.py --config config/reinforce_gamma05.yaml --seed 1 --run-name reinforce-gamma0.5-seed1 --wandb-mode online
python train.py --config config/reinforce_gamma05.yaml --seed 2 --run-name reinforce-gamma0.5-seed2 --wandb-mode online
python train.py --config config/reinforce_gamma05.yaml --seed 3 --run-name reinforce-gamma0.5-seed3 --wandb-mode online
python train.py --config config/reinforce_gamma05.yaml --seed 4 --run-name reinforce-gamma0.5-seed4 --wandb-mode online
```

生成 5-seed 统计表：

```powershell
python eval_seeds.py --config config/reinforce_gamma05.yaml
```

## Actor-Critic

Actor-Critic 是必做算法，也需要跑两个 gamma，每个 gamma 跑 5 个 seed。

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

## PPO

PPO 是进阶选做算法。若小组选择 PPO，除 gamma 对比外，还需要做“去掉 clip”的消融实验。

### PPO, gamma=0.95

```powershell
python train.py --config config/ppo.yaml --gamma 0.95 --seed 0 --run-name ppo-gamma0.95-seed0 --wandb-mode online
python train.py --config config/ppo.yaml --gamma 0.95 --seed 1 --run-name ppo-gamma0.95-seed1 --wandb-mode online
python train.py --config config/ppo.yaml --gamma 0.95 --seed 2 --run-name ppo-gamma0.95-seed2 --wandb-mode online
python train.py --config config/ppo.yaml --gamma 0.95 --seed 3 --run-name ppo-gamma0.95-seed3 --wandb-mode online
python train.py --config config/ppo.yaml --gamma 0.95 --seed 4 --run-name ppo-gamma0.95-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/ppo.yaml --gamma 0.95
```

### PPO, gamma=0.5

```powershell
python train.py --config config/ppo.yaml --gamma 0.5 --seed 0 --run-name ppo-gamma0.5-seed0 --wandb-mode online
python train.py --config config/ppo.yaml --gamma 0.5 --seed 1 --run-name ppo-gamma0.5-seed1 --wandb-mode online
python train.py --config config/ppo.yaml --gamma 0.5 --seed 2 --run-name ppo-gamma0.5-seed2 --wandb-mode online
python train.py --config config/ppo.yaml --gamma 0.5 --seed 3 --run-name ppo-gamma0.5-seed3 --wandb-mode online
python train.py --config config/ppo.yaml --gamma 0.5 --seed 4 --run-name ppo-gamma0.5-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/ppo.yaml --gamma 0.5
```

### PPO 消融：去掉 clip, gamma=0.95

```powershell
python train.py --config config/ppo_no_clip.yaml --seed 0 --run-name ppo-gamma0.95-seed0-no_clip --wandb-mode online
python train.py --config config/ppo_no_clip.yaml --seed 1 --run-name ppo-gamma0.95-seed1-no_clip --wandb-mode online
python train.py --config config/ppo_no_clip.yaml --seed 2 --run-name ppo-gamma0.95-seed2-no_clip --wandb-mode online
python train.py --config config/ppo_no_clip.yaml --seed 3 --run-name ppo-gamma0.95-seed3-no_clip --wandb-mode online
python train.py --config config/ppo_no_clip.yaml --seed 4 --run-name ppo-gamma0.95-seed4-no_clip --wandb-mode online
```

```powershell
python eval_seeds.py --config config/ppo_no_clip.yaml
```

## DDPG

DDPG 是进阶选做算法。若小组选择 DDPG，除 gamma 对比外，还需要做“去掉 target critic”的消融实验。

### DDPG, gamma=0.95

```powershell
python train.py --config config/ddpg.yaml --gamma 0.95 --seed 0 --run-name ddpg-gamma0.95-seed0 --wandb-mode online
python train.py --config config/ddpg.yaml --gamma 0.95 --seed 1 --run-name ddpg-gamma0.95-seed1 --wandb-mode online
python train.py --config config/ddpg.yaml --gamma 0.95 --seed 2 --run-name ddpg-gamma0.95-seed2 --wandb-mode online
python train.py --config config/ddpg.yaml --gamma 0.95 --seed 3 --run-name ddpg-gamma0.95-seed3 --wandb-mode online
python train.py --config config/ddpg.yaml --gamma 0.95 --seed 4 --run-name ddpg-gamma0.95-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/ddpg.yaml --gamma 0.95
```

### DDPG, gamma=0.5

```powershell
python train.py --config config/ddpg.yaml --gamma 0.5 --seed 0 --run-name ddpg-gamma0.5-seed0 --wandb-mode online
python train.py --config config/ddpg.yaml --gamma 0.5 --seed 1 --run-name ddpg-gamma0.5-seed1 --wandb-mode online
python train.py --config config/ddpg.yaml --gamma 0.5 --seed 2 --run-name ddpg-gamma0.5-seed2 --wandb-mode online
python train.py --config config/ddpg.yaml --gamma 0.5 --seed 3 --run-name ddpg-gamma0.5-seed3 --wandb-mode online
python train.py --config config/ddpg.yaml --gamma 0.5 --seed 4 --run-name ddpg-gamma0.5-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/ddpg.yaml --gamma 0.5
```

### DDPG 消融：去掉 target critic, gamma=0.95

```powershell
python train.py --config config/ddpg_no_target_critic.yaml --seed 0 --run-name ddpg-gamma0.95-seed0-no_target_critic --wandb-mode online
python train.py --config config/ddpg_no_target_critic.yaml --seed 1 --run-name ddpg-gamma0.95-seed1-no_target_critic --wandb-mode online
python train.py --config config/ddpg_no_target_critic.yaml --seed 2 --run-name ddpg-gamma0.95-seed2-no_target_critic --wandb-mode online
python train.py --config config/ddpg_no_target_critic.yaml --seed 3 --run-name ddpg-gamma0.95-seed3-no_target_critic --wandb-mode online
python train.py --config config/ddpg_no_target_critic.yaml --seed 4 --run-name ddpg-gamma0.95-seed4-no_target_critic --wandb-mode online
```

```powershell
python eval_seeds.py --config config/ddpg_no_target_critic.yaml
```

## SAC

SAC 是进阶选做算法。若小组选择 SAC，除 gamma 对比外，还需要做“固定 alpha，去掉自适应熵系数”的消融实验。

### SAC, gamma=0.95

```powershell
python train.py --config config/sac.yaml --gamma 0.95 --seed 0 --run-name sac-gamma0.95-seed0 --wandb-mode online
python train.py --config config/sac.yaml --gamma 0.95 --seed 1 --run-name sac-gamma0.95-seed1 --wandb-mode online
python train.py --config config/sac.yaml --gamma 0.95 --seed 2 --run-name sac-gamma0.95-seed2 --wandb-mode online
python train.py --config config/sac.yaml --gamma 0.95 --seed 3 --run-name sac-gamma0.95-seed3 --wandb-mode online
python train.py --config config/sac.yaml --gamma 0.95 --seed 4 --run-name sac-gamma0.95-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/sac.yaml --gamma 0.95
```

### SAC, gamma=0.5

```powershell
python train.py --config config/sac.yaml --gamma 0.5 --seed 0 --run-name sac-gamma0.5-seed0 --wandb-mode online
python train.py --config config/sac.yaml --gamma 0.5 --seed 1 --run-name sac-gamma0.5-seed1 --wandb-mode online
python train.py --config config/sac.yaml --gamma 0.5 --seed 2 --run-name sac-gamma0.5-seed2 --wandb-mode online
python train.py --config config/sac.yaml --gamma 0.5 --seed 3 --run-name sac-gamma0.5-seed3 --wandb-mode online
python train.py --config config/sac.yaml --gamma 0.5 --seed 4 --run-name sac-gamma0.5-seed4 --wandb-mode online
```

```powershell
python eval_seeds.py --config config/sac.yaml --gamma 0.5
```

### SAC 消融：固定 alpha, gamma=0.95

```powershell
python train.py --config config/sac_fixed_alpha.yaml --seed 0 --run-name sac-gamma0.95-seed0-fixed_alpha --wandb-mode online
python train.py --config config/sac_fixed_alpha.yaml --seed 1 --run-name sac-gamma0.95-seed1-fixed_alpha --wandb-mode online
python train.py --config config/sac_fixed_alpha.yaml --seed 2 --run-name sac-gamma0.95-seed2-fixed_alpha --wandb-mode online
python train.py --config config/sac_fixed_alpha.yaml --seed 3 --run-name sac-gamma0.95-seed3-fixed_alpha --wandb-mode online
python train.py --config config/sac_fixed_alpha.yaml --seed 4 --run-name sac-gamma0.95-seed4-fixed_alpha --wandb-mode online
```

```powershell
python eval_seeds.py --config config/sac_fixed_alpha.yaml
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
