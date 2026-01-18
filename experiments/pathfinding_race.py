"""
Pathfinding Race Experiment: Two agents compete to reach point B from point A.
Visualizes agents moving on the map.
"""

import numpy as np
import sys
import os
from typing import Tuple, List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.pathfinding_game import PathfindingGame
from agents.pathfinding_qlearning import PathfindingQLearningAgent
from agents.pathfinding_greedy import PathfindingGreedyAgent
from analysis.pathfinding_visualizer import PathfindingVisualizer


class PathfindingRace:
    """
    Race experiment: Two agents try to reach goal as fast as possible.
    """
    
    def __init__(self, game: PathfindingGame, agent1, agent2, max_steps: int = 1000):
        """
        Initialize pathfinding race.
        
        Args:
            game: PathfindingGame instance
            agent1: First agent (must have act() and update() methods)
            agent2: Second agent (must have act() and update() methods)
            max_steps: Maximum steps per episode
        """
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_steps = max_steps
    
    def run_episode(self, visualize: bool = False) -> Dict:
        """
        Run a single episode race.
        
        Args:
            visualize: Whether to visualize the episode
            
        Returns:
            Dictionary with results:
            - agent1_positions: List of positions
            - agent2_positions: List of positions
            - agent1_rewards: List of rewards
            - agent2_rewards: List of rewards
            - agent1_done: Whether agent 1 reached goal
            - agent2_done: Whether agent 2 reached goal
            - agent1_steps: Steps taken by agent 1
            - agent2_steps: Steps taken by agent 2
        """
        # Reset agents
        self.agent1.reset()
        self.agent2.reset()
        
        # Initialize positions
        pos1 = self.game.start_pos
        pos2 = self.game.start_pos  # Both start at same position
        
        # Track history
        positions1 = [pos1]
        positions2 = [pos2]
        rewards1 = []
        rewards2 = []
        
        done1 = False
        done2 = False
        
        for step in range(self.max_steps):
            # Agent 1 action
            if not done1:
                if isinstance(self.agent1, PathfindingQLearningAgent):
                    state1 = self.game.get_state_index(pos1)
                    action1 = self.agent1.act(state1, self.game)
                elif isinstance(self.agent1, PathfindingGreedyAgent):
                    action1 = self.agent1.act(pos1, self.game.goal_pos, self.game)
                else:
                    # Generic agent interface
                    action1 = self.agent1.act(self.game)
                
                new_pos1, reward1, done1 = self.game.move(pos1, action1)
                
                # Update agent 1
                if isinstance(self.agent1, PathfindingQLearningAgent):
                    state1 = self.game.get_state_index(pos1)
                    next_state1 = self.game.get_state_index(new_pos1) if not done1 else None
                    self.agent1.update(state1, action1, reward1, next_state1, done1)
                else:
                    self.agent1.update(action1, reward1)
                
                pos1 = new_pos1
                positions1.append(pos1)
                rewards1.append(reward1)
            
            # Agent 2 action
            if not done2:
                if isinstance(self.agent2, PathfindingQLearningAgent):
                    state2 = self.game.get_state_index(pos2)
                    action2 = self.agent2.act(state2, self.game)
                elif isinstance(self.agent2, PathfindingGreedyAgent):
                    action2 = self.agent2.act(pos2, self.game.goal_pos, self.game)
                else:
                    action2 = self.agent2.act(self.game)
                
                new_pos2, reward2, done2 = self.game.move(pos2, action2)
                
                # Update agent 2
                if isinstance(self.agent2, PathfindingQLearningAgent):
                    state2 = self.game.get_state_index(pos2)
                    next_state2 = self.game.get_state_index(new_pos2) if not done2 else None
                    self.agent2.update(state2, action2, reward2, next_state2, done2)
                else:
                    self.agent2.update(action2, reward2)
                
                pos2 = new_pos2
                positions2.append(pos2)
                rewards2.append(reward2)
            
            # Both done or max steps reached
            if (done1 and done2) or step >= self.max_steps - 1:
                break
        
        results = {
            'agent1_positions': positions1,
            'agent2_positions': positions2,
            'agent1_rewards': rewards1,
            'agent2_rewards': rewards2,
            'agent1_done': done1,
            'agent2_done': done2,
            'agent1_steps': len(positions1) - 1,
            'agent2_steps': len(positions2) - 1,
        }
        
        # Visualize if requested
        if visualize:
            visualizer = PathfindingVisualizer(self.game)
            visualizer.visualize_episode(
                positions1, positions2,
                agent1_name=self.agent1.name,
                agent2_name=self.agent2.name,
                interval=200,
                show_path=True
            )
        
        return results
    
    def run_multiple_episodes(self, n_episodes: int = 10, visualize_last: bool = True) -> Dict:
        """
        Run multiple episodes and collect statistics.
        
        Args:
            n_episodes: Number of episodes to run
            visualize_last: Whether to visualize the last episode
            
        Returns:
            Dictionary with aggregated results
        """
        all_results = []
        agent1_wins = 0
        agent2_wins = 0
        ties = 0
        
        for episode in range(n_episodes):
            visualize = (episode == n_episodes - 1) and visualize_last
            results = self.run_episode(visualize=visualize)
            all_results.append(results)
            
            # Count wins
            steps1 = results['agent1_steps']
            steps2 = results['agent2_steps']
            
            if results['agent1_done'] and results['agent2_done']:
                if steps1 < steps2:
                    agent1_wins += 1
                elif steps2 < steps1:
                    agent2_wins += 1
                else:
                    ties += 1
            elif results['agent1_done']:
                agent1_wins += 1
            elif results['agent2_done']:
                agent2_wins += 1
        
        # Aggregate statistics
        avg_steps1 = np.mean([r['agent1_steps'] for r in all_results])
        avg_steps2 = np.mean([r['agent2_steps'] for r in all_results])
        success_rate1 = np.mean([r['agent1_done'] for r in all_results])
        success_rate2 = np.mean([r['agent2_done'] for r in all_results])
        
        summary = {
            'n_episodes': n_episodes,
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'ties': ties,
            'avg_steps_agent1': avg_steps1,
            'avg_steps_agent2': avg_steps2,
            'success_rate_agent1': success_rate1,
            'success_rate_agent2': success_rate2,
            'all_results': all_results,
        }
        
        return summary


def main():
    """Run pathfinding race experiment."""
    print("=" * 60)
    print("PATHFINDING RACE: Q-Learning vs Greedy Agent")
    print("=" * 60)
    
    # Create game - CHOOSE ONE:
    print("\nCreating game...")
    
    # Option 1: Simple maze
    # game = PathfindingGame.create_simple_maze(size=12)
    
    # Option 2: Hard maze
    # game = PathfindingGame.create_hard_maze(size=15)
    
    # Option 3: ATHENS MAP! 🏛️
    game = PathfindingGame.create_from_athens_map(
        image_path="athens_map.png",
        size=(50, 50),
        start_pos=(5, 5),  # Adjust based on map
        goal_pos=(45, 45)  # Adjust based on map
    )
    
    n_states = game.height * game.width
    
    print(f"Map size: {game.height}x{game.width}")
    print(f"Start: {game.start_pos}, Goal: {game.goal_pos}")
    
    # Optimal path length
    optimal_steps = game.get_optimal_path_length()
    if optimal_steps:
        print(f"Optimal path length: {optimal_steps} steps")
    
    # Create agents
    print("\nCreating agents...")
    agent1 = PathfindingQLearningAgent(
        n_states=n_states,
        n_actions=4,
        learning_rate=0.1,
        epsilon=0.2,  # Start with more exploration
        discount=0.99,  # Higher discount (future rewards more important)
        epsilon_decay=0.995,  # Gradually reduce exploration
        min_epsilon=0.05,  # Minimum exploration
        name="Q-Learning"
    )
    
    agent2 = PathfindingGreedyAgent(name="Greedy")
    
    print(f"Agent 1: {agent1.name}")
    print(f"Agent 2: {agent2.name}")
    
    # Create race - max 300 steps per episode
    race = PathfindingRace(game, agent1, agent2, max_steps=300)
    
    # Run multiple episodes for learning
    n_learning_episodes = 50  # Good number for learning
    print(f"\nRunning {n_learning_episodes} episodes for Q-Learning to learn...")
    print("(Will show only first and last episode for comparison)")
    
    # Track all metrics
    all_metrics = []
    
    # First episode (for comparison)
    print("\n" + "="*60)
    print("EPISODE 1 (First Run):")
    print("="*60)
    first_results = race.run_episode(visualize=False)
    first_metrics = {
        'episode': 1,
        'agent1_steps': first_results['agent1_steps'],
        'agent2_steps': first_results['agent2_steps'],
        'agent1_done': first_results['agent1_done'],
        'agent2_done': first_results['agent2_done'],
        'agent1_total_reward': sum(first_results['agent1_rewards']),
        'agent2_total_reward': sum(first_results['agent2_rewards']),
        'agent1_avg_reward': np.mean(first_results['agent1_rewards']) if first_results['agent1_rewards'] else 0,
        'agent2_avg_reward': np.mean(first_results['agent2_rewards']) if first_results['agent2_rewards'] else 0,
    }
    all_metrics.append(first_metrics)
    
    print(f"Q-Learning:")
    print(f"  Steps: {first_metrics['agent1_steps']}")
    print(f"  Reached goal: {first_metrics['agent1_done']}")
    print(f"  Total reward: {first_metrics['agent1_total_reward']:.2f}")
    print(f"  Avg reward per step: {first_metrics['agent1_avg_reward']:.4f}")
    print(f"Greedy:")
    print(f"  Steps: {first_metrics['agent2_steps']}")
    print(f"  Reached goal: {first_metrics['agent2_done']}")
    print(f"  Total reward: {first_metrics['agent2_total_reward']:.2f}")
    print(f"  Avg reward per step: {first_metrics['agent2_avg_reward']:.4f}")
    
    # Learning episodes (silent - no output)
    print(f"\nRunning episodes 2-{n_learning_episodes-1} (learning in progress, no output)...")
    for i in range(1, n_learning_episodes - 1):
        results = race.run_episode(visualize=False)
        metrics = {
            'episode': i + 1,
            'agent1_steps': results['agent1_steps'],
            'agent2_steps': results['agent2_steps'],
            'agent1_done': results['agent1_done'],
            'agent2_done': results['agent2_done'],
            'agent1_total_reward': sum(results['agent1_rewards']),
            'agent2_total_reward': sum(results['agent2_rewards']),
            'agent1_avg_reward': np.mean(results['agent1_rewards']) if results['agent1_rewards'] else 0,
            'agent2_avg_reward': np.mean(results['agent2_rewards']) if results['agent2_rewards'] else 0,
        }
        all_metrics.append(metrics)
    
    # Last episode (for comparison)
    print(f"\nRunning episode {n_learning_episodes} (final run with visualization)...")
    print("="*60)
    print(f"EPISODE {n_learning_episodes} (Final Run):")
    print("="*60)
    last_results = race.run_episode(visualize=True)
    last_metrics = {
        'episode': n_learning_episodes,
        'agent1_steps': last_results['agent1_steps'],
        'agent2_steps': last_results['agent2_steps'],
        'agent1_done': last_results['agent1_done'],
        'agent2_done': last_results['agent2_done'],
        'agent1_total_reward': sum(last_results['agent1_rewards']),
        'agent2_total_reward': sum(last_results['agent2_rewards']),
        'agent1_avg_reward': np.mean(last_results['agent1_rewards']) if last_results['agent1_rewards'] else 0,
        'agent2_avg_reward': np.mean(last_results['agent2_rewards']) if last_results['agent2_rewards'] else 0,
    }
    all_metrics.append(last_metrics)
    
    print(f"Q-Learning:")
    print(f"  Steps: {last_metrics['agent1_steps']}")
    print(f"  Reached goal: {last_metrics['agent1_done']}")
    print(f"  Total reward: {last_metrics['agent1_total_reward']:.2f}")
    print(f"  Avg reward per step: {last_metrics['agent1_avg_reward']:.4f}")
    if isinstance(agent1, PathfindingQLearningAgent):
        print(f"  Current epsilon: {agent1.epsilon:.4f} (exploration rate)")
        print(f"  Q-table stats: min={np.min(agent1.Q):.2f}, max={np.max(agent1.Q):.2f}, mean={np.mean(agent1.Q):.2f}")
    print(f"Greedy:")
    print(f"  Steps: {last_metrics['agent2_steps']}")
    print(f"  Reached goal: {last_metrics['agent2_done']}")
    print(f"  Total reward: {last_metrics['agent2_total_reward']:.2f}")
    print(f"  Avg reward per step: {last_metrics['agent2_avg_reward']:.4f}")
    
    # Comparison summary
    print("\n" + "="*60)
    print("COMPARISON: Episode 1 vs Episode " + str(n_learning_episodes))
    print("="*60)
    print(f"Q-Learning:")
    print(f"  Episode 1: {first_metrics['agent1_steps']} steps, Goal reached: {first_metrics['agent1_done']}, Reward: {first_metrics['agent1_total_reward']:.2f}")
    print(f"  Episode {n_learning_episodes}: {last_metrics['agent1_steps']} steps, Goal reached: {last_metrics['agent1_done']}, Reward: {last_metrics['agent1_total_reward']:.2f}")
    if last_metrics['agent1_steps'] < first_metrics['agent1_steps']:
        improvement = first_metrics['agent1_steps'] - last_metrics['agent1_steps']
        print(f"  *** IMPROVEMENT: {improvement} fewer steps! ({((improvement/first_metrics['agent1_steps'])*100):.1f}% better) ***")
    elif last_metrics['agent1_done'] and not first_metrics['agent1_done']:
        print(f"  *** IMPROVEMENT: Now reaches goal (didn't in episode 1)! ***")
    print(f"Greedy (baseline - no change):")
    print(f"  Episode 1: {first_metrics['agent2_steps']} steps, Goal reached: {first_metrics['agent2_done']}")
    print(f"  Episode {n_learning_episodes}: {last_metrics['agent2_steps']} steps, Goal reached: {last_metrics['agent2_done']}")
    
    results = last_results  # For later use
    
    # Full metrics summary
    print("\n" + "="*60)
    print("FULL METRICS SUMMARY (All " + str(n_learning_episodes) + " Episodes):")
    print("="*60)
    
    # Q-Learning stats
    ql_steps = [m['agent1_steps'] for m in all_metrics]
    ql_success = [m['agent1_done'] for m in all_metrics]
    ql_rewards = [m['agent1_total_reward'] for m in all_metrics]
    
    print(f"Q-Learning Statistics:")
    print(f"  Average steps: {np.mean(ql_steps):.1f}")
    print(f"  Min steps: {np.min(ql_steps)}")
    print(f"  Max steps: {np.max(ql_steps)}")
    print(f"  Success rate: {(sum(ql_success)/len(ql_success)*100):.1f}% ({sum(ql_success)}/{len(ql_success)} episodes)")
    print(f"  Average total reward: {np.mean(ql_rewards):.2f}")
    if sum(ql_success) > 0:
        print(f"  Best episode: Episode {ql_steps.index(min([s for i, s in enumerate(ql_steps) if ql_success[i]]))+1} ({min([s for i, s in enumerate(ql_steps) if ql_success[i]])} steps)")
    
    # Greedy stats (should be constant)
    greedy_steps = [m['agent2_steps'] for m in all_metrics]
    greedy_success = [m['agent2_done'] for m in all_metrics]
    
    print(f"\nGreedy Statistics (baseline):")
    print(f"  Average steps: {np.mean(greedy_steps):.1f}")
    print(f"  Success rate: {(sum(greedy_success)/len(greedy_success)*100):.1f}% ({sum(greedy_success)}/{len(greedy_success)} episodes)")
    
    print("\n" + "=" * 60)
    print("FINAL EPISODE RESULTS:")
    print("=" * 60)
    print(f"Agent 1 ({agent1.name}): {results['agent1_steps']} steps, Reached goal: {results['agent1_done']}")
    print(f"Agent 2 ({agent2.name}): {results['agent2_steps']} steps, Reached goal: {results['agent2_done']}")
    
    if results['agent1_done'] and results['agent2_done']:
        if results['agent1_steps'] < results['agent2_steps']:
            print(f"\n*** WINNER: {agent1.name}! ***")
        elif results['agent2_steps'] < results['agent1_steps']:
            print(f"\n*** WINNER: {agent2.name}! ***")
        else:
            print("\n*** It's a tie! ***")
    elif results['agent1_done']:
        print(f"\n*** WINNER: {agent1.name}! ***")
    elif results['agent2_done']:
        print(f"\n*** WINNER: {agent2.name}! ***")
    
    # Save final paths plot
    print("\nGenerating final paths plot...")
    visualizer = PathfindingVisualizer(game)
    visualizer.plot_final_paths(
        results['agent1_positions'],
        results['agent2_positions'],
        agent1_name=agent1.name,
        agent2_name=agent2.name,
        save_path='results/plots/pathfinding_final_paths.png'
    )
    
    print("\n*** Experiment complete! ***")
    print(f"Final paths plot saved to: results/plots/pathfinding_final_paths.png")


if __name__ == '__main__':
    main()
