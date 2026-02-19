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
    plot_exploitability_heatmap, get_plot_path, ensure_results_dir,
    empirical_strategy_history
)
import numpy as np
import os


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
    
    # Generate plots (use empirical action frequency for RL so plot is interpretable)
    ensure_results_dir()
    base = lambda name: get_plot_path('matching_pennies', 'RL vs RL', name)
    n_actions = 2  # Matching Pennies
    plot_strategy_evolution(
        empirical_strategy_history(results['agent1_action_history'], n_actions),
        game, 'RL1', base('strategy1')
    )
    plot_strategy_evolution(
        empirical_strategy_history(results['agent2_action_history'], n_actions),
        game, 'RL2', base('strategy2')
    )
    plot_distance_to_nash(
        results['agent1_distance_history'],
        'RL1', base('distance1')
    )
    plot_distance_to_nash(
        results['agent2_distance_history'],
        'RL2', base('distance2')
    )
    plot_comparison_multiple_agents(
        {
            'RL1': results['agent1_distance_history'],
            'RL2': results['agent2_distance_history']
        },
        'Distance to Nash', base('distance_comparison')
    )
    plot_comparison_multiple_agents(
        {
            'RL1': np.cumsum(results['agent1_reward_history']),
            'RL2': np.cumsum(results['agent2_reward_history'])
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
        f.write("- Both players use Q-Learning in Matching Pennies.\n")
        f.write("- If both distances to Nash stay large, learning is failing to find the equilibrium.\n")
        f.write("- Reward comparison near 0 indicates no systematic exploitation between RL1 and RL2.\n\n")
        f.write("Plots: strategy1.png, strategy2.png, distance1.png, distance2.png, distance_comparison.png, reward_comparison.png, exploitability_heatmap.png\n")

    print("Plots and results.txt saved to results/plots/matching_pennies/(RL vs RL)/")
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
    
    # Generate plots (use empirical action frequency for RL so plot is interpretable)
    ensure_results_dir()
    base = lambda name: get_plot_path('rock_paper_scissors', 'RL vs RL', name)
    n_actions = 3  # Rock-Paper-Scissors
    plot_strategy_evolution(
        empirical_strategy_history(results['agent1_action_history'], n_actions),
        game, 'RL1', base('strategy1')
    )
    plot_strategy_evolution(
        empirical_strategy_history(results['agent2_action_history'], n_actions),
        game, 'RL2', base('strategy2')
    )
    plot_distance_to_nash(
        results['agent1_distance_history'],
        'RL1', base('distance1')
    )
    plot_distance_to_nash(
        results['agent2_distance_history'],
        'RL2', base('distance2')
    )
    plot_comparison_multiple_agents(
        {
            'RL1': results['agent1_distance_history'],
            'RL2': results['agent2_distance_history']
        },
        'Distance to Nash', base('distance_comparison')
    )
    plot_comparison_multiple_agents(
        {
            'RL1': np.cumsum(results['agent1_reward_history']),
            'RL2': np.cumsum(results['agent2_reward_history'])
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
        f.write("- RL vs RL in Rock-Paper-Scissors should ideally converge to the mixed Nash equilibrium.\n")
        f.write("- Large distances to Nash and strongly positive/negative cumulative rewards indicate cyclic or exploitative dynamics.\n")
        f.write("- Use the strategy plots to see if the policies stabilize or keep cycling between actions.\n\n")
        f.write("Plots: strategy1.png, strategy2.png, distance1.png, distance2.png, distance_comparison.png, reward_comparison.png, exploitability_heatmap.png\n")

    print("Plots and results.txt saved to results/plots/rock_paper_scissors/(RL vs RL)/")
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
    heatmap_path = get_plot_path('matching_pennies', 'RL vs RL', 'exploitability_heatmap')
    plot_exploitability_heatmap(
        exploitability_results, learning_rates, epsilons,
        heatmap_path
    )
    print(f"Heatmap saved to {heatmap_path}")


def run_hyperparameter_sweep_rps():
    """Run hyperparameter sweep for exploitability heatmap (Rock-Paper-Scissors)."""
    print("\n" + "="*60)
    print("RL Hyperparameter Sweep - Exploitability Heatmap (RPS)")
    print("="*60)

    learning_rates = [0.01, 0.1, 0.5]
    epsilons = [0.05, 0.1, 0.2]
    game = RockPaperScissors()

    exploitability_results = np.zeros((len(learning_rates), len(epsilons)))

    for i, lr in enumerate(learning_rates):
        for j, eps in enumerate(epsilons):
            rl = QLearningAgent(3, learning_rate=lr, epsilon=eps, name='RL')
            runner = ExperimentRunner(game, rl, rl, n_iterations=5000, seed=42)
            results = runner.run(verbose=False)

            from analysis.metrics import exploitability
            final_strategy = results['agent1_final_strategy']
            expl = exploitability(final_strategy, game)
            exploitability_results[i, j] = expl

            print(f"lr={lr:.2f}, eps={eps:.2f}: exploitability={expl:.4f}")

    ensure_results_dir()
    heatmap_path = get_plot_path('rock_paper_scissors', 'RL vs RL', 'exploitability_heatmap')
    plot_exploitability_heatmap(
        exploitability_results, learning_rates, epsilons,
        heatmap_path
    )
    print(f"Heatmap saved to {heatmap_path}")


if __name__ == '__main__':
    print("Running RL vs RL Experiments...")
    print("This may take a few minutes...\n")
    
    # Run experiments with default hyperparameters
    results_mp = run_rl_vs_rl_matching_pennies(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42)
    results_rps = run_rl_vs_rl_rps(n_iterations=10000, learning_rate=0.1, epsilon=0.1, seed=42)
    
    # Run hyperparameter sweep
    run_hyperparameter_sweep()
    run_hyperparameter_sweep_rps()

    print("\n" + "="*60)
    print("All RL vs RL experiments completed!")
    print("="*60)

