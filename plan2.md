# Project Plan: Fictitious Play & Reinforcement Learning

*(Markdown Outline -- Ready to Use)*

## 1. Title & Abstract

**Title:** *Fictitious Play vs Reinforcement Learning: Experimental
Comparison in Repeated 2‑Player Games*

**Abstract:**\
This project studies and compares two learning mechanisms in repeated
strategic interactions: Fictitious Play (FP) and Reinforcement Learning
(RL). We evaluate their convergence behaviour across three
representative games---Matching Pennies, Prisoner's Dilemma, and
Rock--Paper--Scissors---and analyze their empirical performance,
stability, and distance to Nash equilibria. The study includes
implementation, experimentation, and visualization of agent learning
dynamics.

------------------------------------------------------------------------

## 2. Introduction

-   Motivation: Why is learning in games important?\
-   Applications: multi-agent systems, economics, distributed
    decision-making, AI training.\
-   Goal of the project: Compare FP and RL empirically under a unified
    experimental framework.\
-   Contributions:
    -   Implementation of FP and Q-learning agents.
    -   Evaluation over 3 classic two-player games.
    -   Convergence & stability analysis.
    -   Comparison to theoretical Nash equilibria.

------------------------------------------------------------------------

## 3. Related Work

-   Classical FP literature (Brown 1951, Robinson 1951).
-   Q-learning (Watkins 1989) and multi-agent RL foundations.
-   Foundational material from the course textbook:
    -   **Shoham & Leyton-Brown (2009)** -- *Multiagent Systems*.\
-   Short description of how each method fits into game-theoretic
    learning.

------------------------------------------------------------------------

## 4. Games Studied

### 4.1 Matching Pennies

-   Zero-sum game.
-   Payoff matrix definition.
-   Theoretical mixed Nash equilibrium: (0.5, 0.5).

### 4.2 Prisoner's Dilemma

-   Social dilemma game.
-   Payoff description (T \> R \> P \> S).
-   Dominant strategy: Defect.

### 4.3 Rock--Paper--Scissors

-   Cyclic competition.
-   Payoff logic.
-   Mixed Nash equilibrium: (1/3, 1/3, 1/3).

------------------------------------------------------------------------

## 5. Methodology -- Fictitious Play

-   FP assumptions:
    -   Each agent best-responds to empirical frequencies of opponent
        actions.
-   Algorithm steps:
    1.  Initialize beliefs (uniform recommended).
    2.  At each step, compute best response to empirical distribution.
    3.  Update empirical frequencies.
-   Termination criteria:
    -   Max iterations OR
    -   Frequency change \< threshold.
-   Metrics collected: strategy evolution, distance to Nash, payoff
    sequences.

------------------------------------------------------------------------

## 6. Methodology -- Reinforcement Learning (Q‑Learning)

-   RL setup for repeated matrix games.
-   Definitions:
    -   State = previous step or stateless setup (normal-form).
    -   Actions = available pure strategies.
    -   Rewards = payoffs from the game.
-   Q-learning update rule.
-   Exploration strategy:
    -   ε-greedy exploration.
    -   ε schedules (constant or decaying).
-   Hyperparameters:
    -   Learning rate α
    -   Discount γ (optional for repeated games)
    -   ε ∈ \[0.05, 0.1\]
-   Training protocol:
    -   Episodes, steps, seeds.
-   Metrics collected:
    -   Reward curves.
    -   Convergence behaviour.
    -   Policy stabilization.

------------------------------------------------------------------------

## 7. Experimental Setup

### 7.1 Configuration

-   Number of runs: 30 seeds per method.
-   Episodes: 5,000 per run.
-   Consistent environment for both FP and RL.
-   Logging format: JSONL for steps, CSV for run summaries.

### 7.2 Evaluation Metrics

-   Action frequency over time.
-   Average reward over time.
-   Distance to Nash (L1 norm).
-   Convergence time.
-   Variance across seeds.
-   Statistical tests (optional: t-test, bootstrap).

### 7.3 Files Produced

-   Logs: `/data/logs/*.jsonl`
-   Summaries: `/results/tables/*.csv`
-   Plots: `/results/plots/*.png`

------------------------------------------------------------------------

## 8. Results

### 8.1 Matching Pennies

-   FP: cycles around mixed Nash.
-   RL: converges toward 0.5--0.5 distribution.
-   Plots:
    -   Action frequency trajectory.
    -   Reward curve.
    -   Distribution across seeds.

### 8.2 Prisoner's Dilemma

-   FP: converges rapidly to (Defect, Defect).
-   RL: similar convergence but with exploration deviations.
-   Plots:
    -   Heatmap of final strategies.
    -   Learning curve.

### 8.3 Rock--Paper--Scissors

-   FP: exhibits persistent cycles.
-   RL: converges near (1/3,1/3,1/3) depending on parameters.
-   Plots:
    -   Simplex trajectory.
    -   Frequency evolution.

------------------------------------------------------------------------

## 9. Discussion

-   FP: theoretically elegant, fast in some games, but cycles in
    non-potential games.
-   RL: more flexible, empirical convergence, sensitive to
    hyperparameters.
-   Comparison:
    -   Stability, convergence speed, Nash proximity.
    -   Interpretability vs adaptability.

------------------------------------------------------------------------

## 10. Conclusions

-   Summary of key findings.
-   When FP works better.
-   When RL is more effective.
-   High-level takeaways for multi-agent learning.

------------------------------------------------------------------------

## 11. Future Work

-   Multi-agent RL in larger games.
-   Deep RL (DQN, policy gradients).
-   Stochastic & dynamic games.
-   Learning in extensive-form games.

------------------------------------------------------------------------

## Appendix

-   Full payoff matrices.
-   Hyperparameter tables.
-   Extra plots.
-   Raw logs description.
