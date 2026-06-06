# RL Project
<<<<<<< Updated upstream
强化学习课程第二组代码管理
=======

Policy-gradient comparison project for the course lab. The lab document uses
`CartPole-v0` and requires `5000` training episodes.

## Structure

```text
RL Project/
  train.py                 # unified training entrypoint
  eval_seeds.py            # 5-seed final-100-episode summary
  plot_results.py          # reward curve plotting
  run_sweep.py             # local multi-algorithm runner
  algorithms/
    reinforce.py
    actor_critic.py
    ppo.py
    ddpg.py
    sac.py
  config/
    reinforce.yaml
    reinforce_gamma095.yaml
    reinforce_gamma05.yaml
    actor_critic.yaml
    ppo.yaml
    ddpg.yaml
    sac.yaml
```

## Install

Use the existing conda environment; do not change packages unless necessary.

```bash
conda activate rl_env
```

## Experiment Run Checklist

The lab document requires `CartPole-v0`, `5000` episodes, gamma comparison
experiments with `gamma=0.95` and `gamma=0.5`, and a 5-seed summary table.

`python train.py --algo reinforce` is only a short default example. For formal
experiments, prefer the config-based commands below.

### REINFORCE

Run the main training experiment and the two required gamma ablations:

```bash
python train.py --config config/reinforce.yaml
python train.py --config config/reinforce_gamma095.yaml
python train.py --config config/reinforce_gamma05.yaml
```

Then run the 5-seed summary:

```bash
python eval_seeds.py --config config/reinforce.yaml
```

```bash
python train.py --config config/reinforce_gamma095.yaml --seed 0 --run-name reinforce-gamma0.95-seed0
python train.py --config config/reinforce_gamma095.yaml --seed 1 --run-name reinforce-gamma0.95-seed1
python train.py --config config/reinforce_gamma095.yaml --seed 2 --run-name reinforce-gamma0.95-seed2
python train.py --config config/reinforce_gamma095.yaml --seed 3 --run-name reinforce-gamma0.95-seed3
python train.py --config config/reinforce_gamma095.yaml --seed 4 --run-name reinforce-gamma0.95-seed4

python train.py --config config/reinforce_gamma05.yaml --seed 0 --run-name reinforce-gamma0.5-seed0
python train.py --config config/reinforce_gamma05.yaml --seed 1 --run-name reinforce-gamma0.5-seed1
python train.py --config config/reinforce_gamma05.yaml --seed 2 --run-name reinforce-gamma0.5-seed2
python train.py --config config/reinforce_gamma05.yaml --seed 3 --run-name reinforce-gamma0.5-seed3
python train.py --config config/reinforce_gamma05.yaml --seed 4 --run-name reinforce-gamma0.5-seed4
```

### Actor-Critic

Run the main training experiment and the two required gamma ablations:

```bash
python train.py --config config/actor_critic.yaml
python train.py --config config/actor_critic.yaml --gamma 0.95 --run-name actor-critic-gamma095
python train.py --config config/actor_critic.yaml --gamma 0.5 --run-name actor-critic-gamma05
```

Then run the 5-seed summary:

```bash
python eval_seeds.py --config config/actor_critic.yaml
```

### PPO

Run the main training experiment and the two required gamma ablations:

```bash
python train.py --config config/ppo.yaml
python train.py --config config/ppo.yaml --gamma 0.95 --run-name ppo-gamma095
python train.py --config config/ppo.yaml --gamma 0.5 --run-name ppo-gamma05
```

Then run the 5-seed summary:

```bash
python eval_seeds.py --config config/ppo.yaml
```

PPO is an optional advanced algorithm. If your group selects PPO, the lab also
requires an extra ablation experiment, such as removing PPO clipping and
comparing the result with the normal PPO run.

### DDPG

Run the main training experiment and the two required gamma ablations:

```bash
python train.py --config config/ddpg.yaml
python train.py --config config/ddpg.yaml --gamma 0.95 --run-name ddpg-gamma095
python train.py --config config/ddpg.yaml --gamma 0.5 --run-name ddpg-gamma05
```

Then run the 5-seed summary:

```bash
python eval_seeds.py --config config/ddpg.yaml
```

DDPG is an optional advanced algorithm. If your group selects DDPG, the lab also
requires an extra ablation experiment, such as removing the target network or
removing action exploration noise and comparing the result with the normal DDPG
run.

### SAC

Run the main training experiment and the two required gamma ablations:

```bash
python train.py --config config/sac.yaml
python train.py --config config/sac.yaml --gamma 0.95 --run-name sac-gamma095
python train.py --config config/sac.yaml --gamma 0.5 --run-name sac-gamma05
```

Then run the 5-seed summary:

```bash
python eval_seeds.py --config config/sac.yaml
```

SAC is an optional advanced algorithm. If your group selects SAC, the lab also
requires an extra ablation experiment, such as removing twin Q-networks or using
a fixed entropy coefficient and comparing the result with the normal SAC run.

## Plot Reward Curves

`train.py` appends metrics to `results/metrics.jsonl`. The plotter groups runs
by `run_name`, `algo`, `gamma`, and `seed`, so multiple algorithms can appear on
the same reward figure.

```bash
python plot_results.py --metrics results/metrics.jsonl --out results/reward_curve.png
```

Use `--raw` to plot raw per-episode reward instead of the 100-episode moving
average.

Training also logs `eval_reward` every 100 episodes by default. Plot the test
curve with:

```bash
python plot_results.py --metrics results/metrics.jsonl --metric eval_reward --out results/eval_curve.png
```

## 5-Seed Evaluation

The representative reward is the average reward of the final 100 training
episodes for each seed. The script reports Reward_max, Reward_average, and
Reward_variance, then writes a CSV under `results/`.

```bash
python eval_seeds.py --config config/reinforce.yaml
python eval_seeds.py --config config/actor_critic.yaml --seeds 0 1 2 3 4
```
>>>>>>> Stashed changes
