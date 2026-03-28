# DQN MountainCar-v0

Solving OpenAI Gymnasium's **MountainCar-v0** with a DQN agent and custom reward shaping, built on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Problem

MountainCar-v0 is a classic hard-exploration environment. The car must build momentum by rocking back and forth before it can reach the flag. A vanilla DQN with the sparse default reward (`-1` per step) almost never discovers a successful trajectory, converging to `-200` every episode.

## Solution — Reward Shaping

A `gym.RewardWrapper` augments the reward signal at training time:

| Condition | Bonus |
|---|---|
| Position ≥ 0.45 (goal) | +100 |
| Position ≥ 0.30 | +10 |
| Position > 0.10 | +3 |
| Any step | `+velocity × 5` |

The velocity term directly encourages momentum-building. Shaping is **only applied during training**; evaluation uses the raw environment.

## Results

| Phase | Reward |
|---|---|
| Early training (ep 1–25) | ~−200 (stuck) |
| Late training (ep ~2880+) | ~+100 (shaped) |
| Eval — raw env (5 episodes) | −140, −159, −167, −137, −152 |

Mean eval reward: **~−151**. A random policy scores −200; consistent sub-−160 performance means the agent is reliably reaching the goal well before the 200-step cutoff.

## Hyperparameters

```python
learning_rate         = 5e-4
gamma                 = 0.99
buffer_size           = 100_000
learning_starts       = 1_000
batch_size            = 128
train_freq            = 4
target_update_interval= 1_000
exploration_fraction  = 0.5
exploration_final_eps = 0.05
total_timesteps       = 500_000
```

## Stack

- Python 3.x
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
