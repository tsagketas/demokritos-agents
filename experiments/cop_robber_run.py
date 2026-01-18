
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.cop_robber_game import CopRobberGame
from agents.cop_robber_agents import GreedyCopAgent, GreedyRobberAgent
from analysis.pathfinding_visualizer import PathfindingVisualizer

def run_cop_robber_episode(game, cop_agent, robber_agent, visualize=False, save_path=None):
    state = game.reset()
    cop_pos, robber_pos = state
    
    cop_history = [cop_pos]
    robber_history = [robber_pos]
    
    done = False
    info = ""
    total_steps = 0
    
    while not done:
        # Get actions
        cop_action = cop_agent.act(state, game)
        robber_action = robber_agent.act(state, game)
        
        # Step
        state, rewards, done, info = game.step(cop_action, robber_action)
        cop_pos, robber_pos = state
        
        cop_history.append(cop_pos)
        robber_history.append(robber_pos)
        total_steps += 1
    
    print(f"Episode finished in {total_steps} steps. Reason: {info}")
    print(f"Final Rewards: Cop {rewards[0]}, Robber {rewards[1]}")
    
    if visualize:
        print("Visualizing... Close window to continue.")
        viz = PathfindingVisualizer(game, save_path=save_path)
        # Hack: Pass Cop as agent1, Robber as agent2
        viz.visualize_episode(
            cop_history, 
            robber_history,
            agent1_name="Cop (Blue)",
            agent2_name="Robber (Purple)",
            interval=100,
            show_path=True
        )

def main():
    print("Initializing Cop and Robber on Athens Map...")
    
    # 1. Load Map (Athens)
    # We use the factory method from PathfindingGame but cast it to CopRobberGame
    # Or just use the map grid directly.
    from games.pathfinding_game import PathfindingGame
    
    # Create temp game to load map
    temp_game = PathfindingGame.create_from_athens_map("athens_map.png", size=(50, 50))
    
    # Define Start Positions (Far apart)
    cop_start = temp_game.start_pos
    robber_start = temp_game.goal_pos # Put robber at the 'goal'
    
    # 2. Create Real Game
    game = CopRobberGame(
        temp_game.map_grid, 
        cop_start=cop_start, 
        robber_start=robber_start,
        max_steps=200,
        slip_prob=0.1  # 10% stochasticity
    )
    
    print(f"Map Size: {game.width}x{game.height}")
    print(f"Cop Start: {cop_start}")
    print(f"Robber Start: {robber_start}")
    
    # 3. Create Agents
    cop = GreedyCopAgent()
    robber = GreedyRobberAgent()
    
    # 4. Run Experiment
    print("\nRunning Pursuit-Evasion...")
    run_cop_robber_episode(game, cop, robber, visualize=True, save_path="results/plots/cop_robber_demo.gif")

if __name__ == "__main__":
    main()
