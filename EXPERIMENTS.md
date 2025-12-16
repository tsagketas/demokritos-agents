# 🧪 Running Experiments

This guide explains how to run the experiments for the Fictitious Play & Reinforcement Learning project.

## 🐳 Docker Commands

### Prerequisites
Make sure Docker and Docker Compose are installed and running.

### Build the Docker Image
```bash
docker-compose build
```

### Run Experiments

#### 1. FP vs FP (Fictitious Play vs Fictitious Play)
```bash
docker-compose run --rm game-learning python -m experiments.fp_vs_fp
```
**Output:** Plots saved to `results/plots/`
- Strategy evolution for both agents
- Distance to Nash comparison
- Cumulative reward comparison
- For both Matching Pennies and Rock-Paper-Scissors

#### 2. RL vs RL (Q-Learning vs Q-Learning)
```bash
docker-compose run --rm game-learning python -m experiments.rl_vs_rl
```
**Output:** Plots saved to `results/plots/`
- Strategy evolution
- Distance to Nash
- Cumulative reward
- Exploitability heatmap (hyperparameter sweep)
- For both Matching Pennies and Rock-Paper-Scissors

#### 3. FP vs RL (Cross-play)
```bash
docker-compose run --rm game-learning python -m experiments.fp_vs_rl
```
**Output:** Plots saved to `results/plots/`
- Strategy evolution for both agents
- Distance to Nash comparison
- Cumulative reward comparison
- Average payoff comparison
- For both Matching Pennies and Rock-Paper-Scissors

### Run All Experiments
```bash
# Run all experiments sequentially
docker-compose run --rm game-learning python -m experiments.fp_vs_fp
docker-compose run --rm game-learning python -m experiments.rl_vs_rl
docker-compose run --rm game-learning python -m experiments.fp_vs_rl
```

### Interactive Shell
```bash
docker-compose run --rm game-learning bash
```

### View Results
After running experiments, check the generated plots:
```bash
# On Windows (PowerShell)
ls results/plots/

# On Linux/Mac
ls results/plots/
```

## 📊 Expected Runtime

- **FP vs FP**: ~2-3 minutes (10,000 iterations × 2 games)
- **RL vs RL**: ~5-7 minutes (10,000 iterations × 2 games + hyperparameter sweep)
- **FP vs RL**: ~2-3 minutes (10,000 iterations × 2 games)

**Total**: ~10-15 minutes for all experiments

## 📁 Generated Files

All plots are saved to `results/plots/` with descriptive names:
- `fp_vs_fp_mp_*.png` - FP vs FP on Matching Pennies
- `fp_vs_fp_rps_*.png` - FP vs FP on Rock-Paper-Scissors
- `rl_vs_rl_mp_*.png` - RL vs RL on Matching Pennies
- `rl_vs_rl_rps_*.png` - RL vs RL on Rock-Paper-Scissors
- `fp_vs_rl_mp_*.png` - FP vs RL on Matching Pennies
- `fp_vs_rl_rps_*.png` - FP vs RL on Rock-Paper-Scissors
- `rl_exploitability_heatmap.png` - Hyperparameter sensitivity

## 🔧 Customization

You can modify experiment parameters in each experiment file:
- `n_iterations`: Number of game iterations (default: 10,000)
- `learning_rate`: Q-Learning learning rate (default: 0.1)
- `epsilon`: Q-Learning exploration rate (default: 0.1)
- `seed`: Random seed for reproducibility (default: 42)

## 📝 Notes

- Experiments use `tqdm` for progress bars
- All plots are saved automatically (no display needed)
- Results are reproducible with fixed seeds
- Plots use publication-ready styling (seaborn + matplotlib)

