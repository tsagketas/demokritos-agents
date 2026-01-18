# ♊ GEMINI_GUIDE.md
**Project Context & Operating Manual**

## 📅 Current Status
**Date:** January 18, 2026
**Phase:** Refactoring & Enhancement
**Goal:** Comparison of Fictitious Play (FP) and Reinforcement Learning (RL) in Zero-Sum Games.

## 🧠 Memory & Conventions

### 1. Project Architecture
- **Games:** Inherit from `BaseGame`. Zero-sum.
  - *Current:* Matching Pennies, RPS.
  - *Planned:* Stochastic Game (Grid World).
- **Agents:** Inherit from `BaseAgent`.
  - *FP:* Belief-based, Best Response.
  - *RL:* Q-Learning (Table-based).
- **Experiments:** `runner.py` orchestrates matches.
- **Analysis:** Metrics in `metrics.py`, plotting in `visualizer.py`.

### 2. Critical Rules
- **Player Roles:** Always distinguish between Row Player (P1, maximizing) and Column Player (P2, minimizing).
  - P1 Payoff Matrix: $A$
  - P2 Payoff Matrix: $-A$ (implied)
- **Performance:** Avoid $O(N)$ operations inside the game loop (10k+ iterations). Use incremental updates.
- **Metrics:** "Distance to Nash" is the gold standard for convergence.
- **Visuals:** Plots must be publication-ready (matplotlib/seaborn).

### 3. File Structure
```
project/
├── games/           # Logic (fix best_response here)
├── agents/          # Algorithms (optimize FP, enhance RL)
├── experiments/     # Runners
├── analysis/        # Metrics & Plots
└── GEMINI_GUIDE.md  # This file
```

## 📝 Usage Guide

### Running Experiments
Use `python -m experiments.<experiment_name>` from the `project` root.
```bash
python -m experiments.fp_vs_fp
python -m experiments.rl_vs_rl
python -m experiments.fp_vs_rl
```

### Adding a New Game
1. Create `games/new_game.py`.
2. Inherit `BaseGame`.
3. Define `payoff_matrix` and `get_nash_equilibrium`.
4. Ensure strictly zero-sum or handle general sum explicitly.

## 📋 The "New" Plan (2026 Edition)

| Step | Task | Status | Priority |
|------|------|--------|----------|
| 1 | **Refactor BaseGame** <br> Fix `best_response` for P2 (Column). | 🔴 Todo | High |
| 2 | **Optimize FP** <br> Incremental belief updates. | 🔴 Todo | High |
| 3 | **Enhance Q-Learning** <br> Add decay for `epsilon` & `lr`. | 🔴 Todo | Medium |
| 4 | **Fix Metrics** <br> Correct `regret` calculation. | 🔴 Todo | Medium |
| 5 | **New Feature: Stochastic Game** <br> Implement a simple state-based game. | ⚪ Todo | Low |
| 6 | **Re-run Experiments** <br> Generate new plots with fixed logic. | ⚪ Todo | High |

---
*Use this file to recall the plan and rules. Update it when significant changes occur.*
