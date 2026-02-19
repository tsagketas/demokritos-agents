# Fictitious Play & Reinforcement Learning

## Εντολές (Python)

```bash
# FP vs FP
python -m experiments.fp_vs_fp

# RL vs RL
python -m experiments.rl_vs_rl

# FP vs RL
python -m experiments.fp_vs_rl

# Hunter–Prey GIFs
python -m analysis.gif_maker
```

## Εντολές (Docker)

```bash
# FP vs FP
docker-compose run --rm game-learning python -m experiments.fp_vs_fp

# RL vs RL
docker-compose run --rm game-learning python -m experiments.rl_vs_rl

# FP vs RL
docker-compose run --rm game-learning python -m experiments.fp_vs_rl

# Hunter–Prey GIFs
docker-compose run --rm game-learning python -m analysis.gif_maker
```
