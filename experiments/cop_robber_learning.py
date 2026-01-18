import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.cop_robber_game import CopRobberGame
from games.pathfinding_game import PathfindingGame
from agents.pathfinding_qlearning import PathfindingQLearningAgent
from analysis.pathfinding_visualizer import PathfindingVisualizer

class RandomAgent:
    def act(self, state, game):
        return np.random.randint(4)
    def update(self, *args):
        pass
    def reset(self):
        pass

def run_episode(game, cop, robber, visualize=False, save_path=None):
    """Run a single episode and optionally visualize it."""
    state_vals = game.reset() # (cop_pos, rob_pos)
    state_idx = game.get_state_index_combined()
    
    done = False
    steps = 0
    
    cop_history = [state_vals[0]]
    robber_history = [state_vals[1]]
    rewards_sum = 0
    
    while not done:
        # Cop Act
        action_cop = cop.act(state_idx, game)
        action_rob = robber.act(state_vals, game)
        
        # Step
        next_state_vals, rewards, done, info = game.step(action_cop, action_rob)
        cop_reward, rob_reward = rewards
        
        # Track history
        cop_history.append(next_state_vals[0])
        robber_history.append(next_state_vals[1])
        rewards_sum += cop_reward
        
        # Get next state index
        next_state_idx = game.get_state_index_combined()
        
        # Update Cop
        cop.update(state_idx, action_cop, cop_reward, next_state_idx, done)
        
        state_idx = next_state_idx
        state_vals = next_state_vals
        steps += 1
        
        if done and visualize:
            print(f"Episode done in {steps} steps. Info: {info}")

    if visualize and save_path:
        viz = PathfindingVisualizer(game, save_path=save_path)
        viz.visualize_episode(
            cop_history, 
            robber_history,
            agent1_name="Cop (Blue)",
            agent2_name="Robber (Purple)",
            interval=100,
            show_path=True
        )
        
    return {
        'steps': steps,
        'info': info,
        'cop_reward': rewards_sum,
        'cop_history': cop_history,
        'robber_history': robber_history
    }

def train_cop():
    print("="*60)
    print("TRAINING Q-LEARNING COP (6x6 Grid)")
    print("="*60)
    
    # 1. Create Small Map (6x6)
    game = CopRobberGame(
        map_grid=np.zeros((6, 6)),  # Empty room
        cop_start=(0, 0),
        robber_start=(5, 5),
        max_steps=50,
        slip_prob=0.1
    )
    
    # 2. Setup Agents
    n_states = (game.width * game.height) ** 2
    
    cop = PathfindingQLearningAgent(
        n_states=n_states,
        n_actions=4,
        learning_rate=0.1,
        epsilon=1.0,      # Start with 100% exploration
        min_epsilon=0.05, 
        epsilon_decay=0.9999, # Slower decay for 50k episodes
        name="QL-Cop"
    )
    
    robber = RandomAgent()
    
    # 3. Training Loop
    n_episodes = 50000
    win_history = []
    
    best_steps = float('inf')
    best_episode_data = None
    best_episode_idx = -1
    
    print(f"State space size: {n_states}")
    print(f"Training for {n_episodes} episodes...")
    
    # --- EPISODE 1 ---
    print("\nVisualizing Episode 1...")
    res = run_episode(game, cop, robber, visualize=True, save_path="results/plots/cop_episode_1.gif")
    win_history.append(1 if res['info'] == "caught" else 0)
    
    # --- TRAINING LOOP ---
    print("Running training episodes...")
    for ep in tqdm(range(2, n_episodes)): # Run up to n-1
        res = run_episode(game, cop, robber, visualize=False)
        
        caught = (res['info'] == "caught")
        win_history.append(1 if caught else 0)
        
        # Track Best Run
        if caught and res['steps'] < best_steps:
            best_steps = res['steps']
            best_episode_data = res
            best_episode_idx = ep

    # --- FINAL EPISODE ---
    print(f"\nVisualizing Episode {n_episodes}...")
    res = run_episode(game, cop, robber, visualize=True, save_path=f"results/plots/cop_episode_{n_episodes}.gif")
    win_history.append(1 if res['info'] == "caught" else 0)
    
    # Check if final was best
    if res['info'] == "caught" and res['steps'] < best_steps:
        best_steps = res['steps']
        best_episode_data = res
        best_episode_idx = n_episodes

    # --- VISUALIZE BEST ---
    if best_episode_data:
        print(f"\nVisualizing Best Episode (Ep {best_episode_idx}, {best_steps} steps)...")
        viz = PathfindingVisualizer(game, save_path="results/plots/cop_best_run.gif")
        viz.visualize_episode(
            best_episode_data['cop_history'], 
            best_episode_data['robber_history'],
            agent1_name="Cop (Best)",
            agent2_name="Robber",
            interval=100,
            show_path=True
        )

    # 4. Results
    window = 1000
    rolling_avg = [np.mean(win_history[max(0, i-window):i+1]) for i in range(len(win_history))]
    
    print(f"\nFinal Win Rate (last {window}): {rolling_avg[-1]*100:.1f}%")
    print(f"Best Catch: {best_steps} steps")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_avg)
    plt.title("Q-Learning Cop Learning Curve (vs Random Robber)")
    plt.ylabel(f"Win Rate (Rolling Avg {window})")
    plt.xlabel("Episode")
    plt.grid(True)
    plt.savefig("results/plots/cop_learning_curve.png")
    print("Saved learning curve to results/plots/cop_learning_curve.png")

if __name__ == "__main__":
    train_cop()
