import numpy as np
from tqdm import tqdm
from analysis.metrics import (
    distance_to_nash, exploitability, cumulative_reward,
    strategy_variance, convergence_speed, strategy_stability
)


class ExperimentRunner:
    """
    Unified runner for both simultaneous matrix games and turn-based stochastic games.
    
    Automatically detects game type:
    - Matrix games: Have get_payoff(action1, action2) and get_nash_equilibrium()
    - Turn-based games: Have step(action) and get_current_player()
    """
    
    def __init__(self, game, agent1, agent2, n_iterations=10000, seed=None):
        """
        Initialize experiment runner.
        
        Args:
            game: Game object (matrix game or turn-based game)
            agent1: First agent (row player / player 0)
            agent2: Second agent (column player / player 1)
            n_iterations: Number of iterations to run
            seed: Random seed for reproducibility
        """
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.n_iterations = n_iterations
        
        if seed is not None:
            np.random.seed(seed)
        
        # Detect game type
        self.is_turn_based = (
            hasattr(game, 'step') and 
            hasattr(game, 'get_current_player') and
            hasattr(game, 'reset')
        )
        self.is_matrix_game = (
            hasattr(game, 'get_payoff') and
            hasattr(game, 'get_nash_equilibrium')
        )
        
        if not self.is_turn_based and not self.is_matrix_game:
            raise ValueError("Game must be either turn-based (has step/get_current_player) or matrix game (has get_payoff/get_nash_equilibrium)")
        
        # Results storage
        self.results = {
            'agent1_strategy_history': [],
            'agent2_strategy_history': [],
            'agent1_distance_history': [],
            'agent2_distance_history': [],
            'agent1_exploitability_history': [],
            'agent2_exploitability_history': [],
            'agent1_reward_history': [],
            'agent2_reward_history': [],
            'agent1_action_history': [],
            'agent2_action_history': [],
        }
        
        # Turn-based specific results
        if self.is_turn_based:
            self.results['n_episodes'] = 0
            self.results['captures'] = []
    
    def run(self, verbose=True):
        """
        Run the experiment.
        
        Args:
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with results
        """
        # Set player IDs
        self.agent1.set_player_id(0)
        self.agent2.set_player_id(1)
        
        # Reset agents
        self.agent1.reset()
        self.agent2.reset()
        
        # Reset game if turn-based
        if self.is_turn_based:
            self.game.reset()
        
        # Get Nash equilibrium for matrix games (for metrics)
        nash = None
        if self.is_matrix_game:
            try:
                nash = self.game.get_nash_equilibrium()
            except (AttributeError, NotImplementedError):
                nash = None
        
        iterator = tqdm(range(self.n_iterations), desc="Running experiment") if verbose else range(self.n_iterations)
        
        for iteration in iterator:
            if self.is_turn_based:
                self._run_turn_based_iteration()
            else:
                self._run_simultaneous_iteration(nash)
        
        # Calculate final metrics
        self._calculate_final_metrics(nash)
        
        return self.results
    
    def _run_simultaneous_iteration(self, nash):
        """Run one iteration for simultaneous matrix games."""
        # Agents choose actions simultaneously
        action1 = self.agent1.act(self.game)
        action2 = self.agent2.act(self.game)
        
        # Get payoffs for zero-sum matrix game
        payoff1 = self.game.get_payoff(action1, action2)
        payoff2 = -payoff1  # Zero-sum: P2 payoff is negative of P1 payoff
        
        # Update agents
        self.agent1.update(action1, payoff1, action2)
        self.agent2.update(action2, payoff2, action1)
        
        # Get current strategies
        strategy1 = self.agent1.get_strategy()
        strategy2 = self.agent2.get_strategy()
        
        # Calculate metrics (only for matrix games)
        if nash is not None:
            dist1 = distance_to_nash(strategy1, nash)
            dist2 = distance_to_nash(strategy2, nash)
            expl1 = exploitability(strategy1, self.game, player_id=0)
            expl2 = exploitability(strategy2, self.game, player_id=1)
        else:
            dist1 = dist2 = expl1 = expl2 = None
        
        # Store results
        self.results['agent1_strategy_history'].append(strategy1.copy())
        self.results['agent2_strategy_history'].append(strategy2.copy())
        if dist1 is not None:
            self.results['agent1_distance_history'].append(dist1)
            self.results['agent2_distance_history'].append(dist2)
            self.results['agent1_exploitability_history'].append(expl1)
            self.results['agent2_exploitability_history'].append(expl2)
        self.results['agent1_reward_history'].append(payoff1)
        self.results['agent2_reward_history'].append(payoff2)
        self.results['agent1_action_history'].append(action1)
        self.results['agent2_action_history'].append(action2)
    
    def _run_turn_based_iteration(self):
        """Run one iteration for turn-based games."""
        # Get current state and player
        state = self.game.get_state() if hasattr(self.game, 'get_state') else None
        current_player = self.game.get_current_player()
        
        # Current player chooses action
        if current_player == 0:
            agent = self.agent1
            opponent_agent = self.agent2
        else:
            agent = self.agent2
            opponent_agent = self.agent1
        
        # Agent acts
        if state is not None:
            action = agent.act(self.game, state=state)
        else:
            action = agent.act(self.game)
        
        # Game step
        reward, done, next_player = self.game.step(action)
        
        # Get next state for Q-learning updates
        next_state = self.game.get_state() if hasattr(self.game, 'get_state') else None
        
        # Update agent (with state info if available)
        if hasattr(agent, 'update') and state is not None:
            # Stochastic Q-Learning needs next_state and done
            agent.update(action, reward, next_state=next_state, done=done)
        else:
            # Standard update
            agent.update(action, reward)
        
        # Store results
        if current_player == 0:
            self.results['agent1_reward_history'].append(reward)
            self.results['agent2_reward_history'].append(0)  # Other player didn't move
            self.results['agent1_action_history'].append(action)
            self.results['agent2_action_history'].append(None)
        else:
            self.results['agent1_reward_history'].append(0)  # Other player didn't move
            self.results['agent2_reward_history'].append(reward)
            self.results['agent1_action_history'].append(None)
            self.results['agent2_action_history'].append(action)
        
        # Track captures (for GridGame)
        if done and hasattr(self.game, 'hunter_pos') and current_player == 0:
            if reward == 10:  # Capture reward
                self.results['captures'].append(tuple(self.game.hunter_pos))
        
        # Reset game if episode ended
        if done:
            self.results['n_episodes'] += 1
            self.game.reset()
    
    def _calculate_final_metrics(self, nash):
        """Calculate final metrics after experiment."""
        # Strategy histories
        if self.results['agent1_strategy_history']:
            self.results['agent1_final_strategy'] = self.results['agent1_strategy_history'][-1]
        if self.results['agent2_strategy_history']:
            self.results['agent2_final_strategy'] = self.results['agent2_strategy_history'][-1]
        
        # Convergence metrics (only for matrix games)
        if nash is not None and self.results['agent1_distance_history']:
            self.results['agent1_convergence'] = convergence_speed(
                self.results['agent1_strategy_history'], nash, threshold=0.05
            )
            self.results['agent2_convergence'] = convergence_speed(
                self.results['agent2_strategy_history'], nash, threshold=0.05
            )
        
        # Cumulative rewards
        self.results['agent1_cumulative_reward'] = cumulative_reward(self.results['agent1_reward_history'])
        self.results['agent2_cumulative_reward'] = cumulative_reward(self.results['agent2_reward_history'])
    
    def get_summary(self):
        """Get summary statistics of the experiment."""
        summary = {
            'agent1_name': self.agent1.name,
            'agent2_name': self.agent2.name,
            'game': self.game.__class__.__name__,
            'game_type': 'turn-based' if self.is_turn_based else 'simultaneous',
            'n_iterations': self.n_iterations,
            'agent1_cumulative_reward': self.results.get('agent1_cumulative_reward', 0),
            'agent2_cumulative_reward': self.results.get('agent2_cumulative_reward', 0),
            'agent1_avg_reward': np.mean(self.results['agent1_reward_history']) if self.results['agent1_reward_history'] else 0,
            'agent2_avg_reward': np.mean(self.results['agent2_reward_history']) if self.results['agent2_reward_history'] else 0,
        }
        
        # Matrix game specific metrics
        if self.is_matrix_game:
            try:
                nash = self.game.get_nash_equilibrium()
                summary['agent1_final_distance'] = self.results['agent1_distance_history'][-1] if self.results['agent1_distance_history'] else None
                summary['agent2_final_distance'] = self.results['agent2_distance_history'][-1] if self.results['agent2_distance_history'] else None
                summary['agent1_convergence'] = self.results.get('agent1_convergence')
                summary['agent2_convergence'] = self.results.get('agent2_convergence')
            except (AttributeError, NotImplementedError):
                pass
        
        # Turn-based game specific metrics
        if self.is_turn_based:
            summary['n_episodes'] = self.results.get('n_episodes', 0)
            summary['n_captures'] = len(self.results.get('captures', []))
            if summary['n_episodes'] > 0:
                summary['capture_rate'] = (summary['n_captures'] / summary['n_episodes']) * 100
            else:
                summary['capture_rate'] = 0
        
        return summary
    
    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print(f"Experiment Summary: {summary['agent1_name']} vs {summary['agent2_name']}")
        print(f"Game: {summary['game']} ({summary['game_type']})")
        print(f"Iterations: {summary['n_iterations']}")
        print("-"*60)
        
        # Matrix game metrics
        if self.is_matrix_game:
            print(f"{summary['agent1_name']}:")
            if 'agent1_final_distance' in summary and summary['agent1_final_distance'] is not None:
                print(f"  Final distance to Nash: {summary['agent1_final_distance']:.4f}")
            if 'agent1_convergence' in summary and summary['agent1_convergence'] is not None:
                print(f"  Convergence at iteration: {summary['agent1_convergence']}")
            print(f"  Cumulative reward: {summary['agent1_cumulative_reward']:.2f}")
            print(f"  Average reward: {summary['agent1_avg_reward']:.4f}")
            
            print(f"\n{summary['agent2_name']}:")
            if 'agent2_final_distance' in summary and summary['agent2_final_distance'] is not None:
                print(f"  Final distance to Nash: {summary['agent2_final_distance']:.4f}")
            if 'agent2_convergence' in summary and summary['agent2_convergence'] is not None:
                print(f"  Convergence at iteration: {summary['agent2_convergence']}")
            print(f"  Cumulative reward: {summary['agent2_cumulative_reward']:.2f}")
            print(f"  Average reward: {summary['agent2_avg_reward']:.4f}")
        
        # Turn-based game metrics
        if self.is_turn_based:
            print(f"{summary['agent1_name']}:")
            print(f"  Cumulative reward: {summary['agent1_cumulative_reward']:.2f}")
            print(f"  Average reward: {summary['agent1_avg_reward']:.4f}")
            
            print(f"\n{summary['agent2_name']}:")
            print(f"  Cumulative reward: {summary['agent2_cumulative_reward']:.2f}")
            print(f"  Average reward: {summary['agent2_avg_reward']:.4f}")
            
            if 'n_episodes' in summary:
                print(f"\nEpisodes: {summary['n_episodes']}")
            if 'n_captures' in summary:
                print(f"Captures: {summary['n_captures']}")
            if 'capture_rate' in summary:
                print(f"Capture rate: {summary['capture_rate']:.1f}%")
        
        print("="*60 + "\n")


# ============================================================================
# Convenience functions for Grid Game experiments
# ============================================================================

def run_grid_experiment(n_iterations=200000, first_player=0, output_subdir='RL vs RL', seed=42):
    """
    Convenience function to run Grid Game experiments with visualization.
    
    This function sets up a Grid Game experiment with Stochastic Q-Learning agents
    and generates plots. For more control, use ExperimentRunner directly.
    
    Args:
        n_iterations: Number of turns to run
        first_player: 0 = Hunter starts, 1 = Prey starts
        output_subdir: Subdirectory name for results
        seed: Random seed
        
    Returns:
        Output directory path
    """
    from games.grid_game import GridGame
    from agents.stochastic_q_learning import StochasticQLearningAgent
    from analysis.visualizer import ensure_results_dir
    import matplotlib.pyplot as plt
    import os
    
    GRID_PLOTS_BASE = os.path.join('results', 'plots', 'grid_game')
    out_dir = os.path.join(GRID_PLOTS_BASE, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    label = "Hunter first" if first_player == 0 else "Prey first"
    print("=" * 60)
    print(f"Grid Game (Hunter vs Prey) - Turn-Based 3x3, max_steps=20, {label}")
    print("=" * 60)

    game = GridGame(size=3, max_steps=20, first_player=first_player)
    n_states = 81

    hunter = StochasticQLearningAgent(
        n_actions=5,
        n_states=n_states,
        learning_rate=0.1,
        epsilon=0.5,
        lr_decay=0.99999,
        epsilon_decay=0.99998,
        discount_factor=0.99,
        name="Hunter"
    )
    prey = StochasticQLearningAgent(
        n_actions=5,
        n_states=n_states,
        learning_rate=0.1,
        epsilon=0.5,
        lr_decay=0.99999,
        epsilon_decay=0.99998,
        discount_factor=0.99,
        name="Prey"
    )

    # Use unified ExperimentRunner
    runner = ExperimentRunner(game, hunter, prey, n_iterations=n_iterations, seed=seed)
    results = runner.run(verbose=True)
    runner.print_summary()

    # Extract results
    n_episodes = results.get('n_episodes', 0)
    captures = results.get('captures', [])
    rewards_hunter = [r for r in results['agent1_reward_history'] if r != 0]
    step_rewards_hunter = results['agent1_reward_history']
    step_rewards_prey = results['agent2_reward_history']

    n_hunter_turns = len(rewards_hunter)
    n_captures = len(captures)
    total_reward_hunter = sum(rewards_hunter)
    mean_reward_hunter = np.mean(rewards_hunter) if rewards_hunter else 0.0

    ensure_results_dir()

    # Txt summary in same folder as plots
    results_txt = os.path.join(out_dir, 'results.txt')
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write(f"Grid Game (Hunter vs Prey) - Turn-Based 3x3, max_steps=20\n")
        f.write(f"First player: {'Hunter' if first_player == 0 else 'Prey'}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Iterations (turns):     {n_iterations}\n")
        f.write(f"Episodes (games):       {n_episodes}\n")
        f.write(f"Hunter turns:           {n_hunter_turns}\n")
        f.write(f"Captures (Hunter wins): {n_captures}\n")
        capture_rate = (n_captures / n_episodes * 100) if n_episodes else 0
        f.write(f"Capture rate:           {capture_rate:.1f}%\n")
        f.write(f"Hunter total reward:    {total_reward_hunter:.2f}\n")
        f.write(f"Hunter mean reward:     {mean_reward_hunter:.4f}\n")
        f.write(f"Hunter final epsilon:   {hunter.epsilon:.6f}\n")
        f.write(f"Prey final epsilon:     {prey.epsilon:.6f}\n\n")
        f.write("Interpretation:\n")
        f.write("- Slope of cumulative reward > 0 means Hunter is winning on average.\n")
        f.write("- More captures = Hunter catching Prey more often.\n")
        f.write("- Mean reward > 0: Hunter doing better; < 0: Prey escaping / timeout more.\n")
        f.write(f"\nPlots in this folder: cumulative_reward.png, avg_reward.png\n")
    print(f"Results summary: {results_txt}")

    # Plots in same folder: cumulative Hunter + Prey on same graph
    cum_h = np.cumsum(step_rewards_hunter)
    cum_p = np.cumsum(step_rewards_prey)
    plt.figure(figsize=(10, 6))
    plt.plot(cum_h, label='Hunter (cumulative)', color='C0')
    plt.plot(cum_p, label='Prey (cumulative)', color='red')
    plt.title(f'Cumulative Reward ({label})')
    plt.xlabel('Turns')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'cumulative_reward.png'))
    plt.close()

    window = 1000
    if len(rewards_hunter) >= window:
        avg_rewards = np.convolve(rewards_hunter, np.ones(window) / window, mode='valid')
        plt.figure(figsize=(10, 6))
        plt.plot(avg_rewards, label='Hunter Avg Reward (1k window)')
        plt.axhline(y=0, color='r', linestyle='--', label='Even Game')
        plt.title(f'Hunter Average Reward over Time ({label})')
        plt.xlabel('Iterations')
        plt.ylabel('Avg Reward per Step')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'avg_reward.png'))
        plt.close()

    print(f"Plots and results saved in: {out_dir}\n")
    return out_dir


def plot_capture_heatmap(capture_locations, size=3, filename='grid_capture_heatmap.png'):
    """
    Plot heatmap of where captures occurred in Grid Game.
    
    Args:
        capture_locations: List of (row, col) tuples where captures occurred
        size: Grid size (default 3 for 3x3 grid)
        filename: Output filename for the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    grid = np.zeros((size, size))
    for r, c in capture_locations:
        grid[r, c] += 1
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, annot=True, fmt='g', cmap='Reds')
    plt.title(f'Capture Heatmap ({len(capture_locations)} captures)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig(filename)
    plt.close()


# Allow running grid experiments directly from this module
if __name__ == '__main__':
    # Example: Run grid game experiments
    run_grid_experiment(
        n_iterations=200000,
        first_player=0,
        output_subdir='RL vs RL - Hunter first'
    )
    run_grid_experiment(
        n_iterations=200000,
        first_player=1,
        output_subdir='RL vs RL - Prey first'
    )

