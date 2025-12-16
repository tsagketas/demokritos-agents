"""
FP vs RL Experiment
Tests Fictitious Play vs Q-Learning cross-play.
"""

from experiments.runner import ExperimentRunner
from agents.fictitious_play import FictitiousPlayAgent
from agents.q_learning import QLearningAgent
from games.matching_pennies import MatchingPennies
from games.rps import RockPaperScissors
from analysis.visualizer import (
    plot_strategy_evolution, plot_distance_to_nash,
    plot_cumulative_reward, plot_comparison_multiple_agents,
    plot_average_payoff_comparison, ensure_results_dir
)
import numpy as np


def run_fp_vs_rl_matching_pennies(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42):
    """Run FP vs RL on Matching Pennies."""
    print("="*60)
    print(f"FP vs RL - Matching Pennies (RL: lr={learning_rate}, eps={epsilon})")
    print("="*60)
    
    game = MatchingPennies()
    fp = FictitiousPlayAgent(2, 'FP')
    rl = QLearningAgent(2, learning_rate=learning_rate, epsilon=epsilon, name='RL')
    
    runner = ExperimentRunner(game, fp, rl, n_iterations=n_iterations, seed=seed)
    results = runner.run(verbose=True)
    runner.print_summary()
    
    # Generate plots
    ensure_results_dir()
    
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'FP', 'results/plots/fp_vs_rl_mp_strategy_fp.png'
    )
    
    plot_strategy_evolution(
        results['agent2_strategy_history'],
        game, 'RL', 'results/plots/fp_vs_rl_mp_strategy_rl.png'
    )
    
    plot_comparison_multiple_agents(
        {
            'FP': results['agent1_distance_history'],
            'RL': results['agent2_distance_history']
        },
        'Distance to Nash', 'results/plots/fp_vs_rl_mp_distance_comparison.png'
    )
    
    plot_comparison_multiple_agents(
        {
            'FP': np.cumsum(results['agent1_reward_history']),
            'RL': np.cumsum(results['agent2_reward_history'])
        },
        'Cumulative Reward', 'results/plots/fp_vs_rl_mp_reward_comparison.png'
    )
    
    # Average payoff comparison
    from analysis.metrics import average_payoff
    avg_payoffs = {
        'FP': average_payoff(results['agent1_reward_history']),
        'RL': average_payoff(results['agent2_reward_history'])
    }
    plot_average_payoff_comparison(
        avg_payoffs, 'results/plots/fp_vs_rl_mp_avg_payoff.png'
    )
    
    print("Plots saved to results/plots/")
    return results


def run_fp_vs_rl_rps(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42):
    """Run FP vs RL on Rock-Paper-Scissors."""
    print("\n" + "="*60)
    print(f"FP vs RL - Rock-Paper-Scissors (RL: lr={learning_rate}, eps={epsilon})")
    print("="*60)
    
    game = RockPaperScissors()
    fp = FictitiousPlayAgent(3, 'FP')
    rl = QLearningAgent(3, learning_rate=learning_rate, epsilon=epsilon, name='RL')
    
    runner = ExperimentRunner(game, fp, rl, n_iterations=n_iterations, seed=seed)
    results = runner.run(verbose=True)
    runner.print_summary()
    
    # Generate plots
    ensure_results_dir()
    
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'FP', 'results/plots/fp_vs_rl_rps_strategy_fp.png'
    )
    
    plot_strategy_evolution(
        results['agent2_strategy_history'],
        game, 'RL', 'results/plots/fp_vs_rl_rps_strategy_rl.png'
    )
    
    plot_comparison_multiple_agents(
        {
            'FP': results['agent1_distance_history'],
            'RL': results['agent2_distance_history']
        },
        'Distance to Nash', 'results/plots/fp_vs_rl_rps_distance_comparison.png'
    )
    
    print("Plots saved to results/plots/")
    return results


if __name__ == '__main__':
    print("Running FP vs RL Experiments...")
    print("This may take a few minutes...\n")
    
    # Run experiments
    results_mp = run_fp_vs_rl_matching_pennies(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42)
    results_rps = run_fp_vs_rl_rps(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42)
    
    print("\n" + "="*60)
    print("All FP vs RL experiments completed!")
    print("="*60)

