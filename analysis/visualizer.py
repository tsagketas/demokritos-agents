import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def get_plot_path(game_name, agent_combo, plot_name):
    """
    Return path for a plot in the organized structure: results/plots/{game}/({agent_combo})/{plot_name}.png
    Creates the directory if it does not exist.
    """
    dir_path = os.path.join('results', 'plots', game_name, agent_combo)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f'{plot_name}.png')


def plot_strategy_evolution(strategy_history, game, agent_name, save_path=None):
    """
    Plot strategy evolution over time.
    
    Args:
        strategy_history: List of strategies (each is probability distribution)
        game: Game object (for action names)
        agent_name: Name of the agent
        save_path: Optional path to save plot
    """
    strategies = np.array(strategy_history)
    iterations = range(len(strategies))
    
    plt.figure(figsize=(10, 6))
    for action_idx in range(strategies.shape[1]):
        action_name = game.get_action_name(action_idx) if hasattr(game, 'get_action_name') else f'Action {action_idx}'
        plt.plot(iterations, strategies[:, action_idx], label=action_name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Strategy Evolution: {agent_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_distance_to_nash(distance_history, agent_name, save_path=None):
    """
    Plot distance to Nash equilibrium over time.
    
    Args:
        distance_history: List of distances to Nash
        agent_name: Name of the agent
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(distance_history, linewidth=2, color='darkblue')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Distance to Nash (L2)', fontsize=12)
    plt.title(f'Distance to Nash: {agent_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_cumulative_reward(reward_history, agent_name, save_path=None):
    """
    Plot cumulative reward over time.
    
    Args:
        reward_history: List of rewards
        agent_name: Name of the agent
        save_path: Optional path to save plot
    """
    cumulative = np.cumsum(reward_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative, linewidth=2, color='green')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title(f'Cumulative Reward: {agent_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_exploitability_heatmap(results, learning_rates, epsilons, save_path=None):
    """
    Plot exploitability heatmap for different hyperparameters.
    
    Args:
        results: 2D array of exploitability values (lr x epsilon)
        learning_rates: List of learning rates
        epsilons: List of epsilon values
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, 
                xticklabels=[f'{eps:.2f}' for eps in epsilons],
                yticklabels=[f'{lr:.2f}' for lr in learning_rates],
                annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'Exploitability'})
    plt.xlabel('Epsilon', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Exploitability Heatmap (RL Agent)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_average_payoff_comparison(results_dict, save_path=None):
    """
    Plot bar chart comparing average payoffs.
    
    Args:
        results_dict: Dictionary {agent_name: average_payoff}
        save_path: Optional path to save plot
    """
    agents = list(results_dict.keys())
    payoffs = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agents, payoffs, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel('Average Payoff', fontsize=12)
    plt.title('Average Payoff Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_convergence_speed(convergence_times, agent_names, save_path=None):
    """
    Plot bar chart of convergence speed.
    
    Args:
        convergence_times: List of iterations to converge (or None if didn't converge)
        agent_names: List of agent names
        save_path: Optional path to save plot
    """
    # Replace None with max value for visualization
    max_time = max([t for t in convergence_times if t is not None] + [10000])
    times = [t if t is not None else max_time for t in convergence_times]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agent_names, times, color=['#3498db', '#e74c3c'])
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel('Iterations to 95% Nash', fontsize=12)
    plt.title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_regret_curves(regret_history_dict, save_path=None):
    """
    Plot regret curves for multiple agents.
    
    Args:
        regret_history_dict: Dictionary {agent_name: [regret over time]}
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    for agent_name, regret_history in regret_history_dict.items():
        plt.plot(regret_history, label=agent_name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title('Regret Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_comparison_multiple_agents(metric_history_dict, metric_name, save_path=None):
    """
    Plot comparison of a metric across multiple agents.
    
    Args:
        metric_history_dict: Dictionary {agent_name: [metric values over time]}
        metric_name: Name of the metric (for title/labels)
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    for agent_name, history in metric_history_dict.items():
        plt.plot(history, label=agent_name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def empirical_strategy_history(action_history, n_actions):
    """
    Compute empirical (running) strategy from action history.
    At each step t, returns the distribution of actions played so far.
    Makes strategy evolution plots meaningful for RL agents (instead of
    plotting epsilon-greedy policy which jumps every step).
    """
    strategies = []
    counts = np.zeros(n_actions)
    for a in action_history:
        counts[a] += 1
        strategies.append((counts / counts.sum()).copy())
    return strategies


def ensure_results_dir():
    """Ensure results directories exist (flat and nested structure)."""
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

