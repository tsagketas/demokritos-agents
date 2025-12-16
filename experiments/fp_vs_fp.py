"""
FP vs FP Experiment
Tests Fictitious Play agents playing against each other.
"""

from experiments.runner import ExperimentRunner
from agents.fictitious_play import FictitiousPlayAgent
from games.matching_pennies import MatchingPennies
from games.rps import RockPaperScissors
from analysis.visualizer import (
    plot_strategy_evolution, plot_distance_to_nash,
    plot_cumulative_reward, plot_comparison_multiple_agents,
    ensure_results_dir
)
import numpy as np


def run_fp_vs_fp_matching_pennies(n_iterations=10000, seed=42):
    """Run FP vs FP on Matching Pennies."""
    print("="*60)
    print("FP vs FP - Matching Pennies")
    print("="*60)
    
    game = MatchingPennies()
    fp1 = FictitiousPlayAgent(2, 'FP1')
    fp2 = FictitiousPlayAgent(2, 'FP2')
    
    runner = ExperimentRunner(game, fp1, fp2, n_iterations=n_iterations, seed=seed)
    results = runner.run(verbose=True)
    runner.print_summary()
    
    # Generate plots
    ensure_results_dir()
    
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'FP1', 'results/plots/fp_vs_fp_mp_strategy1.png'
    )
    
    plot_strategy_evolution(
        results['agent2_strategy_history'],
        game, 'FP2', 'results/plots/fp_vs_fp_mp_strategy2.png'
    )
    
    plot_distance_to_nash(
        results['agent1_distance_history'],
        'FP1', 'results/plots/fp_vs_fp_mp_distance1.png'
    )
    
    plot_distance_to_nash(
        results['agent2_distance_history'],
        'FP2', 'results/plots/fp_vs_fp_mp_distance2.png'
    )
    
    plot_comparison_multiple_agents(
        {
            'FP1': results['agent1_distance_history'],
            'FP2': results['agent2_distance_history']
        },
        'Distance to Nash', 'results/plots/fp_vs_fp_mp_distance_comparison.png'
    )
    
    plot_comparison_multiple_agents(
        {
            'FP1': np.cumsum(results['agent1_reward_history']),
            'FP2': np.cumsum(results['agent2_reward_history'])
        },
        'Cumulative Reward', 'results/plots/fp_vs_fp_mp_reward_comparison.png'
    )
    
    print("Plots saved to results/plots/")
    return results


def run_fp_vs_fp_rps(n_iterations=10000, seed=42):
    """Run FP vs FP on Rock-Paper-Scissors."""
    print("\n" + "="*60)
    print("FP vs FP - Rock-Paper-Scissors")
    print("="*60)
    
    game = RockPaperScissors()
    fp1 = FictitiousPlayAgent(3, 'FP1')
    fp2 = FictitiousPlayAgent(3, 'FP2')
    
    runner = ExperimentRunner(game, fp1, fp2, n_iterations=n_iterations, seed=seed)
    results = runner.run(verbose=True)
    runner.print_summary()
    
    # Generate plots
    ensure_results_dir()
    
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'FP1', 'results/plots/fp_vs_fp_rps_strategy1.png'
    )
    
    plot_strategy_evolution(
        results['agent2_strategy_history'],
        game, 'FP2', 'results/plots/fp_vs_fp_rps_strategy2.png'
    )
    
    plot_distance_to_nash(
        results['agent1_distance_history'],
        'FP1', 'results/plots/fp_vs_fp_rps_distance1.png'
    )
    
    plot_comparison_multiple_agents(
        {
            'FP1': results['agent1_distance_history'],
            'FP2': results['agent2_distance_history']
        },
        'Distance to Nash', 'results/plots/fp_vs_fp_rps_distance_comparison.png'
    )
    
    print("Plots saved to results/plots/")
    return results


if __name__ == '__main__':
    print("Running FP vs FP Experiments...")
    print("This may take a few minutes...\n")
    
    # Run experiments
    results_mp = run_fp_vs_fp_matching_pennies(n_iterations=10000, seed=42)
    results_rps = run_fp_vs_fp_rps(n_iterations=10000, seed=42)
    
    print("\n" + "="*60)
    print("All FP vs FP experiments completed!")
    print("="*60)

