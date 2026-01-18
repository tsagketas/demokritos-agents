import numpy as np
from tqdm import tqdm
from analysis.metrics import (
    distance_to_nash, exploitability, cumulative_reward,
    strategy_variance, convergence_speed, strategy_stability
)


class ExperimentRunner:
    """Runner for game experiments between agents."""
    
    def __init__(self, game, agent1, agent2, n_iterations=10000, seed=None):
        """
        Initialize experiment runner.
        
        Args:
            game: Game object
            agent1: First agent (row player)
            agent2: Second agent (column player)
            n_iterations: Number of iterations to run
            seed: Random seed for reproducibility
        """
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.n_iterations = n_iterations
        
        if seed is not None:
            np.random.seed(seed)
        
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
        
        nash = self.game.get_nash_equilibrium()
        
        iterator = tqdm(range(self.n_iterations), desc="Running experiment") if verbose else range(self.n_iterations)
        
        for iteration in iterator:
            # Agents choose actions
            action1 = self.agent1.act(self.game)
            action2 = self.agent2.act(self.game)
            
            # Get payoffs and handle state transitions
            if hasattr(self.game, 'step'):
                payoff1, payoff2 = self.game.step(action1, action2)
            else:
                payoff1 = self.game.get_payoff(action1, action2)
                payoff2 = -payoff1  # Zero-sum: P2 payoff is negative of P1 payoff
            
            # Update agents
            self.agent1.update(action1, payoff1, action2)
            self.agent2.update(action2, payoff2, action1)
            
            # Get current strategies
            strategy1 = self.agent1.get_strategy()
            strategy2 = self.agent2.get_strategy()
            
            # Calculate metrics
            dist1 = distance_to_nash(strategy1, nash)
            dist2 = distance_to_nash(strategy2, nash)
            expl1 = exploitability(strategy1, self.game)
            expl2 = exploitability(strategy2, self.game)
            
            # Store results
            self.results['agent1_strategy_history'].append(strategy1.copy())
            self.results['agent2_strategy_history'].append(strategy2.copy())
            self.results['agent1_distance_history'].append(dist1)
            self.results['agent2_distance_history'].append(dist2)
            self.results['agent1_exploitability_history'].append(expl1)
            self.results['agent2_exploitability_history'].append(expl2)
            self.results['agent1_reward_history'].append(payoff1)
            self.results['agent2_reward_history'].append(payoff2)
            self.results['agent1_action_history'].append(action1)
            self.results['agent2_action_history'].append(action2)
        
        # Calculate final metrics
        self.results['agent1_final_strategy'] = strategy1
        self.results['agent2_final_strategy'] = strategy2
        self.results['agent1_convergence'] = convergence_speed(
            self.results['agent1_strategy_history'], nash, threshold=0.05
        )
        self.results['agent2_convergence'] = convergence_speed(
            self.results['agent2_strategy_history'], nash, threshold=0.05
        )
        self.results['agent1_cumulative_reward'] = cumulative_reward(self.results['agent1_reward_history'])
        self.results['agent2_cumulative_reward'] = cumulative_reward(self.results['agent2_reward_history'])
        
        return self.results
    
    def get_summary(self):
        """Get summary statistics of the experiment."""
        nash = self.game.get_nash_equilibrium()
        
        summary = {
            'agent1_name': self.agent1.name,
            'agent2_name': self.agent2.name,
            'game': self.game.__class__.__name__,
            'n_iterations': self.n_iterations,
            'agent1_final_distance': self.results['agent1_distance_history'][-1] if self.results['agent1_distance_history'] else None,
            'agent2_final_distance': self.results['agent2_distance_history'][-1] if self.results['agent2_distance_history'] else None,
            'agent1_convergence': self.results.get('agent1_convergence'),
            'agent2_convergence': self.results.get('agent2_convergence'),
            'agent1_cumulative_reward': self.results.get('agent1_cumulative_reward', 0),
            'agent2_cumulative_reward': self.results.get('agent2_cumulative_reward', 0),
            'agent1_avg_reward': np.mean(self.results['agent1_reward_history']) if self.results['agent1_reward_history'] else 0,
            'agent2_avg_reward': np.mean(self.results['agent2_reward_history']) if self.results['agent2_reward_history'] else 0,
        }
        
        return summary
    
    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print(f"Experiment Summary: {summary['agent1_name']} vs {summary['agent2_name']}")
        print(f"Game: {summary['game']}")
        print(f"Iterations: {summary['n_iterations']}")
        print("-"*60)
        print(f"{summary['agent1_name']}:")
        print(f"  Final distance to Nash: {summary['agent1_final_distance']:.4f}")
        print(f"  Convergence at iteration: {summary['agent1_convergence']}")
        print(f"  Cumulative reward: {summary['agent1_cumulative_reward']:.2f}")
        print(f"  Average reward: {summary['agent1_avg_reward']:.4f}")
        print(f"\n{summary['agent2_name']}:")
        print(f"  Final distance to Nash: {summary['agent2_final_distance']:.4f}")
        print(f"  Convergence at iteration: {summary['agent2_convergence']}")
        print(f"  Cumulative reward: {summary['agent2_cumulative_reward']:.2f}")
        print(f"  Average reward: {summary['agent2_avg_reward']:.4f}")
        print("="*60 + "\n")

