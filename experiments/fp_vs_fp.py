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
    get_plot_path, ensure_results_dir
)
import numpy as np
import os


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
    base = lambda name: get_plot_path('matching_pennies', 'FP vs FP', name)
    
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'FP1', base('strategy1')
    )
    plot_strategy_evolution(
        results['agent2_strategy_history'],
        game, 'FP2', base('strategy2')
    )
    plot_distance_to_nash(
        results['agent1_distance_history'],
        'FP1', base('distance1')
    )
    plot_distance_to_nash(
        results['agent2_distance_history'],
        'FP2', base('distance2')
    )
    plot_comparison_multiple_agents(
        {
            'FP1': results['agent1_distance_history'],
            'FP2': results['agent2_distance_history']
        },
        'Distance to Nash', base('distance_comparison')
    )
    plot_comparison_multiple_agents(
        {
            'FP1': np.cumsum(results['agent1_reward_history']),
            'FP2': np.cumsum(results['agent2_reward_history'])
        },
        'Cumulative Reward', base('reward_comparison')
    )

    # Save textual summary
    summary = runner.get_summary()
    results_dir = os.path.dirname(base('strategy1'))
    results_path = os.path.join(results_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"{summary['game']}: {summary['agent1_name']} vs {summary['agent2_name']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Iterations: {summary['n_iterations']}\n")
        f.write(f"{summary['agent1_name']} final distance to Nash: {summary['agent1_final_distance']:.4f}\n")
        f.write(f"{summary['agent2_name']} final distance to Nash: {summary['agent2_final_distance']:.4f}\n")
        f.write(f"{summary['agent1_name']} cumulative reward: {summary['agent1_cumulative_reward']:.2f}\n")
        f.write(f"{summary['agent2_name']} cumulative reward: {summary['agent2_cumulative_reward']:.2f}\n")
        f.write(f"{summary['agent1_name']} average reward: {summary['agent1_avg_reward']:.4f}\n")
        f.write(f"{summary['agent2_name']} average reward: {summary['agent2_avg_reward']:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("- Both agents use Fictitious Play in Matching Pennies.\n")
        f.write("- Distances close to 0 mean strategies are near the mixed Nash equilibrium (50-50 heads/tails).\n")
        f.write("- Reward comparison shows whether either FP agent systematically exploits the other.\n\n")
        f.write("Plots: strategy1.png, strategy2.png, distance1.png, distance2.png, distance_comparison.png, reward_comparison.png\n")

    print("Plots and results.txt saved to results/plots/matching_pennies/(FP vs FP)/")
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
    base = lambda name: get_plot_path('rock_paper_scissors', 'FP vs FP', name)
    
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'FP1', base('strategy1')
    )
    plot_strategy_evolution(
        results['agent2_strategy_history'],
        game, 'FP2', base('strategy2')
    )
    plot_distance_to_nash(
        results['agent1_distance_history'],
        'FP1', base('distance1')
    )
    plot_distance_to_nash(
        results['agent2_distance_history'],
        'FP2', base('distance2')
    )
    plot_comparison_multiple_agents(
        {
            'FP1': results['agent1_distance_history'],
            'FP2': results['agent2_distance_history']
        },
        'Distance to Nash', base('distance_comparison')
    )
    plot_comparison_multiple_agents(
        {
            'FP1': np.cumsum(results['agent1_reward_history']),
            'FP2': np.cumsum(results['agent2_reward_history'])
        },
        'Cumulative Reward', base('reward_comparison')
    )

    # Save textual summary
    summary = runner.get_summary()
    results_dir = os.path.dirname(base('strategy1'))
    results_path = os.path.join(results_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"{summary['game']}: {summary['agent1_name']} vs {summary['agent2_name']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Iterations: {summary['n_iterations']}\n")
        f.write(f"{summary['agent1_name']} final distance to Nash: {summary['agent1_final_distance']:.4f}\n")
        f.write(f"{summary['agent2_name']} final distance to Nash: {summary['agent2_final_distance']:.4f}\n")
        f.write(f"{summary['agent1_name']} cumulative reward: {summary['agent1_cumulative_reward']:.2f}\n")
        f.write(f"{summary['agent2_name']} cumulative reward: {summary['agent2_cumulative_reward']:.2f}\n")
        f.write(f"{summary['agent1_name']} average reward: {summary['agent1_avg_reward']:.4f}\n")
        f.write(f"{summary['agent2_name']} average reward: {summary['agent2_avg_reward']:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("- Both agents use Fictitious Play in Rock-Paper-Scissors.\n")
        f.write("- Distances close to 0 mean strategies are near the mixed Nash equilibrium (1/3,1/3,1/3).\n")
        f.write("- Reward comparison should hover near 0 when both agents learn the same mixed strategy.\n\n")
        f.write("Plots: strategy1.png, strategy2.png, distance1.png, distance2.png, distance_comparison.png, reward_comparison.png\n")

    print("Plots and results.txt saved to results/plots/rock_paper_scissors/(FP vs FP)/")
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

