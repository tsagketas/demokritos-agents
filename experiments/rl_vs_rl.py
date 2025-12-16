"""
RL vs RL Experiment
Tests Q-Learning agents playing against each other.
"""

from experiments.runner import ExperimentRunner
from agents.q_learning import QLearningAgent
from games.matching_pennies import MatchingPennies
from games.rps import RockPaperScissors
from analysis.visualizer import (
    plot_strategy_evolution, plot_distance_to_nash,
    plot_cumulative_reward, plot_comparison_multiple_agents,
    plot_exploitability_heatmap, ensure_results_dir
)
import numpy as np


def run_rl_vs_rl_matching_pennies(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42):
    """Run RL vs RL on Matching Pennies."""
    print("="*60)
    print(f"RL vs RL - Matching Pennies (lr={learning_rate}, eps={epsilon})")
    print("="*60)
    
    game = MatchingPennies()
    rl1 = QLearningAgent(2, learning_rate=learning_rate, epsilon=epsilon, name='RL1')
    rl2 = QLearningAgent(2, learning_rate=learning_rate, epsilon=epsilon, name='RL2')
    
    runner = ExperimentRunner(game, rl1, rl2, n_iterations=n_iterations, seed=seed)
    results = runner.run(verbose=True)
    runner.print_summary()
    
    # Generate plots
    ensure_results_dir()
    
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'RL1', f'results/plots/rl_vs_rl_mp_strategy1_lr{learning_rate}_eps{epsilon}.png'
    )
    
    plot_distance_to_nash(
        results['agent1_distance_history'],
        'RL1', f'results/plots/rl_vs_rl_mp_distance1_lr{learning_rate}_eps{epsilon}.png'
    )
    
    plot_comparison_multiple_agents(
        {
            'RL1': results['agent1_distance_history'],
            'RL2': results['agent2_distance_history']
        },
        'Distance to Nash', f'results/plots/rl_vs_rl_mp_distance_comparison_lr{learning_rate}_eps{epsilon}.png'
    )
    
    plot_comparison_multiple_agents(
        {
            'RL1': np.cumsum(results['agent1_reward_history']),
            'RL2': np.cumsum(results['agent2_reward_history'])
        },
        'Cumulative Reward', f'results/plots/rl_vs_rl_mp_reward_comparison_lr{learning_rate}_eps{epsilon}.png'
    )
    
    print("Plots saved to results/plots/")
    return results


def run_rl_vs_rl_rps(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42):
    """Run RL vs RL on Rock-Paper-Scissors."""
    print("\n" + "="*60)
    print(f"RL vs RL - Rock-Paper-Scissors (lr={learning_rate}, eps={epsilon})")
    print("="*60)
    
    game = RockPaperScissors()
    rl1 = QLearningAgent(3, learning_rate=learning_rate, epsilon=epsilon, name='RL1')
    rl2 = QLearningAgent(3, learning_rate=learning_rate, epsilon=epsilon, name='RL2')
    
    runner = ExperimentRunner(game, rl1, rl2, n_iterations=n_iterations, seed=seed)
    results = runner.run(verbose=True)
    runner.print_summary()
    
    # Generate plots
    ensure_results_dir()
    
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'RL1', f'results/plots/rl_vs_rl_rps_strategy1_lr{learning_rate}_eps{epsilon}.png'
    )
    
    plot_distance_to_nash(
        results['agent1_distance_history'],
        'RL1', f'results/plots/rl_vs_rl_rps_distance1_lr{learning_rate}_eps{epsilon}.png'
    )
    
    print("Plots saved to results/plots/")
    return results


def run_hyperparameter_sweep():
    """Run hyperparameter sweep for exploitability heatmap."""
    print("\n" + "="*60)
    print("RL Hyperparameter Sweep - Exploitability Heatmap")
    print("="*60)
    
    learning_rates = [0.01, 0.1, 0.5]
    epsilons = [0.05, 0.1, 0.2]
    game = MatchingPennies()
    
    exploitability_results = np.zeros((len(learning_rates), len(epsilons)))
    
    for i, lr in enumerate(learning_rates):
        for j, eps in enumerate(epsilons):
            rl = QLearningAgent(2, learning_rate=lr, epsilon=eps, name='RL')
            runner = ExperimentRunner(game, rl, rl, n_iterations=5000, seed=42)
            results = runner.run(verbose=False)
            
            # Get final exploitability
            from analysis.metrics import exploitability
            final_strategy = results['agent1_final_strategy']
            expl = exploitability(final_strategy, game)
            exploitability_results[i, j] = expl
            
            print(f"lr={lr:.2f}, eps={eps:.2f}: exploitability={expl:.4f}")
    
    # Plot heatmap
    ensure_results_dir()
    plot_exploitability_heatmap(
        exploitability_results, learning_rates, epsilons,
        'results/plots/rl_exploitability_heatmap.png'
    )
    
    print("Heatmap saved to results/plots/rl_exploitability_heatmap.png")


if __name__ == '__main__':
    print("Running RL vs RL Experiments...")
    print("This may take a few minutes...\n")
    
    # Run experiments with default hyperparameters
    results_mp = run_rl_vs_rl_matching_pennies(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42)
    results_rps = run_rl_vs_rl_rps(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42)
    
    # Run hyperparameter sweep
    run_hyperparameter_sweep()
    
    print("\n" + "="*60)
    print("All RL vs RL experiments completed!")
    print("="*60)

