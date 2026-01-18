import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys
import gc

# Add project root to path
sys.path.append(os.getcwd())

from games.grid_game import GridGame
from agents.stochastic_q_learning import StochasticQLearningAgent

# --- CONFIGURATION ---
GRID_SIZE = 5
N_STATES = GRID_SIZE ** 4
TRAIN_ITERATIONS = 200000 
MAX_STEPS_EPISODE = 50 
# ---------------------

def run_episode(game, hunter, prey, hunter_eps=None, prey_eps=None):
    game.hunter_pos = [0, 0]
    game.prey_pos = [GRID_SIZE-1, GRID_SIZE-1]
    
    hunter_path = [game.hunter_pos.copy()]
    prey_path = [game.prey_pos.copy()]
    
    old_eps_h = hunter.epsilon
    old_eps_p = prey.epsilon
    
    if hunter_eps is not None: hunter.epsilon = hunter_eps
    if prey_eps is not None: prey.epsilon = prey_eps

    max_steps = MAX_STEPS_EPISODE
    caught = False
    
    for _ in range(max_steps):
        action_h = hunter.act(game) 
        action_p = prey.act(game) 

        reward_h, reward_p = game.step(action_h, action_p)
        
        hunter_path.append(game.hunter_pos.copy())
        prey_path.append(game.prey_pos.copy())
        
        if reward_h == 10: 
            caught = True
            break
            
    hunter.epsilon = old_eps_h
    prey.epsilon = old_eps_p
            
    return hunter_path, prey_path, caught

def run_dynamic_random_episode(game, hunter, prey):
    max_retries = 20
    for _ in range(max_retries):
        h_path, p_path, caught = run_episode(game, hunter, prey, hunter_eps=1.0, prey_eps=1.0)
        
        initial_pos = h_path[0]
        moved = False
        check_range = min(len(h_path), 5)
        for i in range(1, check_range):
            if h_path[i] != initial_pos:
                moved = True
                break
        
        if moved:
            return h_path, p_path, caught
            
    return h_path, p_path, caught

def find_best_episode(game, hunter, prey, trials=50):
    """Retries random episodes until one shows movement."""
    best_steps = float('inf')
    best_run = None
    
    for _ in range(trials):
        # Hunter Smart (0.0), Prey Random (1.0)
        h_path, p_path, caught = run_episode(game, hunter, prey, hunter_eps=0.0, prey_eps=1.0)
        
        if caught:
            steps = len(h_path) - 1
            if steps < best_steps:
                best_steps = steps
                best_run = (h_path, p_path, caught)
                
    if best_run:
        return best_run[0], best_run[1], best_run[2], best_steps
    else:
        # Fallback if never caught (rare)
        return h_path, p_path, caught, len(h_path)-1

def create_fancy_gif(hunter_path, prey_path, caught, filename, title):
    fig, ax = plt.subplots(figsize=(6, 7))
    plt.subplots_adjust(bottom=0.15)
    
    board = np.zeros((GRID_SIZE, GRID_SIZE))
    board[::2, ::2] = 1
    board[1::2, 1::2] = 1
    
    def update(frame):
        ax.clear()
        ax.imshow(board, cmap='Greys', vmin=0, vmax=2, extent=[-0.5, GRID_SIZE-0.5, GRID_SIZE-0.5, -0.5])
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\nStep: {frame}", fontsize=14, fontweight='bold', pad=15)
        
        h_pos = hunter_path[frame]
        p_pos = prey_path[frame]
        
        ax.plot(h_pos[1], h_pos[0], 's', color='#ff4444', markeredgecolor='darkred', 
                markeredgewidth=2, markersize=22, label='Hunter')
        
        ax.plot(p_pos[1], p_pos[0], 'o', color='#4488ff', markeredgecolor='navy', 
                markeredgewidth=2, markersize=22, label='Prey')
        
        if caught and frame == len(hunter_path) - 1:
            center = (GRID_SIZE-1) / 2
            ax.text(center, center, 'CAUGHT!', color='red', fontsize=24, fontweight='bold', 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round'))
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), 
                  fancybox=True, shadow=True, ncol=2, fontsize=10)

    frames = len(hunter_path)
    if caught: frames += 4
        
    def frame_gen():
        for i in range(len(hunter_path)): yield i
        if caught:
            for _ in range(4): yield len(hunter_path) - 1

    ani = animation.FuncAnimation(fig, update, frames=frame_gen, interval=600, save_count=frames)
    ani.save(filename, writer='pillow', fps=1.5)
    print(f"Saved: {filename}")
    plt.close('all')
    gc.collect()

def main():
    print(f"Initializing 5x5 Agents (States: {N_STATES})...")
    game = GridGame(size=GRID_SIZE)
    
    hunter = StochasticQLearningAgent(game.n_actions, n_states=N_STATES, name="Hunter", epsilon=0.9, learning_rate=0.2)
    prey = StochasticQLearningAgent(game.n_actions, n_states=N_STATES, name="Prey", epsilon=0.9, learning_rate=0.2)

    print("GIF 1: Random...")
    h_path, p_path, caught = run_dynamic_random_episode(game, hunter, prey)
    create_fancy_gif(h_path, p_path, caught, "results/plots/hunter_prey_early.gif", f"Ep 1: Random ({GRID_SIZE}x{GRID_SIZE})")

    print(f"Training ({TRAIN_ITERATIONS} iters)...")
    for i in range(TRAIN_ITERATIONS):
        ah = hunter.act(game)
        ap = prey.act(game)
        rh, rp = game.step(ah, ap)
        hunter.update(ah, rh)
        prey.update(ap, rp)
        if rh == 10: 
            game.hunter_pos = [0, 0]
            game.prey_pos = [GRID_SIZE-1, GRID_SIZE-1]

    print("Searching for Best Episode (50 trials)...")
    h_path, p_path, caught, steps = find_best_episode(game, hunter, prey, trials=50)
    
    title = f"Best Run: {steps} Steps (Optimal)"
    print(f"Found Best Run: {steps} steps")
    
    create_fancy_gif(h_path, p_path, caught, "results/plots/hunter_prey_best.gif", title)

if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)
    main()