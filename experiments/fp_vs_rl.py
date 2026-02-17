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
    plot_average_payoff_comparison, get_plot_path, ensure_results_dir,
    empirical_strategy_history
)
from analysis.metrics import average_payoff
import numpy as np
import os


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
    base = lambda name: get_plot_path('matching_pennies', 'FP vs RL', name)
    n_actions = 2  # Matching Pennies
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'FP', base('strategy_fp')
    )
    plot_strategy_evolution(
        empirical_strategy_history(results['agent2_action_history'], n_actions),
        game, 'RL', base('strategy_rl')
    )
    plot_comparison_multiple_agents(
        {
            'FP': results['agent1_distance_history'],
            'RL': results['agent2_distance_history']
        },
        'Distance to Nash', base('distance_comparison')
    )
    plot_comparison_multiple_agents(
        {
            'FP': np.cumsum(results['agent1_reward_history']),
            'RL': np.cumsum(results['agent2_reward_history'])
        },
        'Cumulative Reward', base('reward_comparison')
    )
    avg_payoffs = {
        'FP': average_payoff(results['agent1_reward_history']),
        'RL': average_payoff(results['agent2_reward_history'])
    }
    plot_average_payoff_comparison(avg_payoffs, base('avg_payoff'))

    # Save textual summary
    summary = runner.get_summary()
    results_dir = os.path.dirname(base('strategy_fp'))
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
        f.write("- FP plays best response to empirical frequencies; RL learns Q-values.\n")
        f.write("- If RL distance to Nash stays large while FP is near 0, RL has not learned the equilibrium.\n")
        f.write("- Positive cumulative reward for RL means it is exploiting FP in Matching Pennies.\n\n")
        f.write("Plots: strategy_fp.png, strategy_rl.png, distance_comparison.png, reward_comparison.png, avg_payoff.png\n")

    print("Plots and results.txt saved to results/plots/matching_pennies/(FP vs RL)/")
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
    base = lambda name: get_plot_path('rock_paper_scissors', 'FP vs RL', name)
    
    n_actions = 3  # Rock-Paper-Scissors
    plot_strategy_evolution(
        results['agent1_strategy_history'],
        game, 'FP', base('strategy_fp')
    )
    plot_strategy_evolution(
        empirical_strategy_history(results['agent2_action_history'], n_actions),
        game, 'RL', base('strategy_rl')
    )
    plot_comparison_multiple_agents(
        {
            'FP': results['agent1_distance_history'],
            'RL': results['agent2_distance_history']
        },
        'Distance to Nash', base('distance_comparison')
    )
    plot_comparison_multiple_agents(
        {
            'FP': np.cumsum(results['agent1_reward_history']),
            'RL': np.cumsum(results['agent2_reward_history'])
        },
        'Cumulative Reward', base('reward_comparison')
    )
    avg_payoffs = {
        'FP': average_payoff(results['agent1_reward_history']),
        'RL': average_payoff(results['agent2_reward_history'])
    }
    plot_average_payoff_comparison(avg_payoffs, base('avg_payoff'))

    # Save textual summary
    summary = runner.get_summary()
    results_dir = os.path.dirname(base('strategy_fp'))
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
        f.write("- FP vs RL in Rock-Paper-Scissors tests if RL can learn the mixed equilibrium.\n")
        f.write("- Distances near 0 for both agents mean they both approximate (1/3,1/3,1/3).\n")
        f.write("- Average payoff tells you if RL is exploiting or being exploited by FP.\n\n")
        f.write("Plots: strategy_fp.png, strategy_rl.png, distance_comparison.png, reward_comparison.png, avg_payoff.png\n")

    print("Plots and results.txt saved to results/plots/rock_paper_scissors/(FP vs RL)/")
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

