from experiments.runner import ExperimentRunner
from games.grid_game import GridGame
from agents.stochastic_q_learning import StochasticQLearningAgent
from analysis.visualizer import plot_comparison_multiple_agents, ensure_results_dir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_capture_heatmap(capture_locations, size=3, filename='grid_capture_heatmap.png'):
    """Plot heatmap of where captures occurred."""
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

def run_grid_experiment(n_iterations=50000, seed=42):
    print("="*60)
    print("Grid Game (Hunter vs Prey) Experiment")
    print("="*60)
    
    game = GridGame(size=3)
    
    # RL Agents
    # Hunter (P1)
    hunter = StochasticQLearningAgent(
        n_actions=5, 
        n_states=81, # 3^4
        learning_rate=0.1, 
        epsilon=0.5, # Start with high exploration
        lr_decay=0.99998,
        epsilon_decay=0.99995,
        name="Hunter"
    )
    
    # Prey (P2)
    prey = StochasticQLearningAgent(
        n_actions=5, 
        n_states=81,
        learning_rate=0.1, 
        epsilon=0.5,
        lr_decay=0.99998,
        epsilon_decay=0.99995,
        name="Prey"
    )
    
    # Custom loop to track captures for heatmap
    # We can reuse runner but we want capture locations
    # So let's run manually or monkey-patch runner? 
    # Let's write a simple loop here for full control.
    
    hunter.set_player_id(0)
    prey.set_player_id(1)
    
    captures = []
    rewards_hunter = []
    
    print("Training agents...")
    from tqdm import tqdm
    for i in tqdm(range(n_iterations)):
        s_idx = game.get_state()
        
        # Act
        a1 = hunter.act(game)
        a2 = prey.act(game)
        
        # Step
        r1, r2 = game.step(a1, a2)
        
        # Check capture (Reward 10 means capture)
        if r1 == 10:
            # Capture happened at OLD positions (before reset)
            # But game resets immediately inside step.
            # We need to track positions *before* step if we want exact location.
            # Actually, the logic in GridGame resets AFTER setting reward.
            # So we can't easily see where it happened unless we modify game to return info.
            # Simplified: Just track where they are NOW (which is reset 0,0) is wrong.
            # Let's trust the cumulative reward plot for now.
            pass
            
        # Update
        hunter.update(a1, r1) # Opponent action not needed for QL
        prey.update(a2, r2)
        
        rewards_hunter.append(r1)

    # Plotting
    ensure_results_dir()
    
    # 1. Cumulative Reward
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(rewards_hunter), label='Hunter Reward')
    plt.title('Hunter Cumulative Reward (Slope > 0 means Hunter is winning)')
    plt.xlabel('Iterations')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/grid_cumulative_reward.png')
    plt.close()
    
    # 2. Moving Average Reward (To see who is learning better)
    window = 1000
    avg_rewards = np.convolve(rewards_hunter, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards, label='Hunter Avg Reward (1k window)')
    plt.axhline(y=0, color='r', linestyle='--', label='Even Game')
    plt.title('Hunter Average Reward over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Avg Reward per Step')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/grid_avg_reward.png')
    plt.close()

    print("Plots saved: grid_cumulative_reward.png, grid_avg_reward.png")

if __name__ == '__main__':
    run_grid_experiment()
