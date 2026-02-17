"""
MinMax (Hunter) vs RL (Prey) on turn-based Grid Game.
Two runs: Hunter first, Prey first.
"""
from games.grid_game import GridGame
from agents.minimax import MinimaxAgent
from agents.stochastic_q_learning import StochasticQLearningAgent
from analysis.visualizer import ensure_results_dir
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

GRID_PLOTS_BASE = os.path.join('results', 'plots', 'grid_game')


def run_minmax_vs_rl(n_iterations=200000, first_player=0, output_subdir='MinMax vs RL', depth=4):
    """
    Hunter = MinMax, Prey = RL.
    first_player: 0 = Hunter first, 1 = Prey first.
    """
    out_dir = os.path.join(GRID_PLOTS_BASE, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    label = "Hunter first" if first_player == 0 else "Prey first"
    print("=" * 60)
    print(f"Grid Game: MinMax (Hunter) vs RL (Prey), max_steps=20, {label}")
    print("=" * 60)

    game = GridGame(size=3, max_steps=20, first_player=first_player)
    n_states = 81

    hunter = MinimaxAgent(n_actions=5, depth=depth, name="Hunter")
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
    hunter.set_player_id(0)
    prey.set_player_id(1)

    captures = []
    rewards_hunter = []
    step_rewards_hunter = []
    step_rewards_prey = []
    n_episodes = 0

    game.reset()

    for i in tqdm(range(n_iterations), desc=output_subdir):
        s_idx = game.get_state()
        current = game.get_current_player()

        if current == 0:
            action = hunter.act(game)
            reward, done, _ = game.step(action)
            next_s = game.get_state()
            hunter.update(action, reward)
            rewards_hunter.append(reward)
            step_rewards_hunter.append(reward)
            step_rewards_prey.append(0)
            if done and reward == 10:
                captures.append(tuple(game.hunter_pos))
        else:
            action = prey.act(game, state=s_idx)
            reward, done, _ = game.step(action)
            next_s = game.get_state()
            prey.update(action, reward, next_state=next_s, done=done)
            step_rewards_hunter.append(0)
            step_rewards_prey.append(reward)

        if done:
            n_episodes += 1
            game.reset()

    n_hunter_turns = len(rewards_hunter)
    n_captures = len(captures)
    total_reward_hunter = sum(rewards_hunter)
    mean_reward_hunter = np.mean(rewards_hunter) if rewards_hunter else 0.0

    ensure_results_dir()

    results_txt = os.path.join(out_dir, 'results.txt')
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write("Grid Game: MinMax (Hunter) vs RL (Prey), max_steps=20\n")
        f.write(f"First player: {'Hunter' if first_player == 0 else 'Prey'}\n")
        f.write(f"MinMax depth: {depth}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Iterations (turns):     {n_iterations}\n")
        f.write(f"Episodes (games):       {n_episodes}\n")
        f.write(f"Hunter turns:           {n_hunter_turns}\n")
        f.write(f"Captures (Hunter wins): {n_captures}\n")
        capture_rate = (n_captures / n_episodes * 100) if n_episodes else 0
        f.write(f"Capture rate:           {capture_rate:.1f}%\n")
        f.write(f"Hunter total reward:    {total_reward_hunter:.2f}\n")
        f.write(f"Hunter mean reward:     {mean_reward_hunter:.4f}\n")
        f.write(f"Prey final epsilon:     {prey.epsilon:.6f}\n\n")
        f.write("Interpretation:\n")
        f.write("- Hunter = MinMax (optimal), Prey = RL (learning).\n")
        f.write("- High capture rate = MinMax Hunter dominates; RL Prey learns to escape over time if rate drops.\n")
        f.write(f"\nPlots: cumulative_reward.png, avg_reward.png\n")
    print(f"Results summary: {results_txt}")

    cum_h = np.cumsum(step_rewards_hunter)
    cum_p = np.cumsum(step_rewards_prey)
    plt.figure(figsize=(10, 6))
    plt.plot(cum_h, label='Hunter MinMax (cumulative)', color='C0')
    plt.plot(cum_p, label='Prey RL (cumulative)', color='red')
    plt.title(f'MinMax vs RL - Cumulative Reward ({label})')
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
        plt.title(f'MinMax vs RL - Hunter Avg Reward ({label})')
        plt.xlabel('Hunter turns')
        plt.ylabel('Avg Reward per Step')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'avg_reward.png'))
        plt.close()

    print(f"Plots and results saved in: {out_dir}\n")
    return out_dir


if __name__ == '__main__':
    run_minmax_vs_rl(
        n_iterations=200000,
        first_player=0,
        output_subdir='MinMax vs RL - Hunter first',
        depth=6
    )
    run_minmax_vs_rl(
        n_iterations=200000,
        first_player=1,
        output_subdir='MinMax vs RL - Prey first',
        depth=6
    )
