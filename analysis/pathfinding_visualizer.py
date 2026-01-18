"""
Visualizer for pathfinding games.
Shows agents moving on the map with matplotlib animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional
import os


class PathfindingVisualizer:
    """
    Visualizes pathfinding agents moving on a map.
    Creates animated plot showing agent positions over time.
    """
    
    def __init__(self, game, save_path: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            game: PathfindingGame instance
            save_path: Optional path to save animation (e.g., 'results/plots/pathfinding_animation.gif')
        """
        self.game = game
        self.save_path = save_path
        self.fig = None
        self.ax = None
    
    def visualize_episode(self, agent1_positions: List[Tuple[int, int]], 
                         agent2_positions: Optional[List[Tuple[int, int]]] = None,
                         agent1_name: str = "Agent 1",
                         agent2_name: str = "Agent 2",
                         interval: int = 200,
                         show_path: bool = True):
        """
        Visualize a single episode with agent movement.
        
        Args:
            agent1_positions: List of (row, col) positions for agent 1
            agent2_positions: Optional list of positions for agent 2 (None if single agent)
            agent1_name: Name for agent 1
            agent2_name: Name for agent 2
            interval: Animation interval in milliseconds
            show_path: Whether to show path trail
        """
        # Create figure with black background
        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor='black')
        self.ax.set_facecolor('black')
        
        # Plot map (white walls on black background)
        map_grid = self.game.map_grid
        # Invert: walls (1) = white, free (0) = black
        inverted_map = 1 - map_grid  # Now 0=wall (black), 1=free (white)
        self.ax.imshow(inverted_map, cmap='gray', alpha=0.8, origin='upper', vmin=0, vmax=1)
        
        # Mark start and goal
        start_r, start_c = self.game.start_pos
        goal_r, goal_c = self.game.goal_pos
        
        self.ax.plot(start_c, start_r, 'go', markersize=15, label='Start (A)', markeredgecolor='darkgreen', markeredgewidth=2)
        self.ax.plot(goal_c, goal_r, 'r*', markersize=20, label='Goal (B)', markeredgecolor='darkred', markeredgewidth=2)
        
        # Prepare animation data
        max_steps = len(agent1_positions)
        if agent2_positions:
            max_steps = max(max_steps, len(agent2_positions))
        
        # Initialize agent markers
        agent1_marker, = self.ax.plot([], [], 'bo', markersize=12, label=agent1_name, markeredgecolor='darkblue', markeredgewidth=2)
        agent1_path = None
        if show_path:
            agent1_path, = self.ax.plot([], [], 'b-', alpha=0.3, linewidth=2)
        
        agent2_marker = None
        agent2_path = None
        if agent2_positions:
            agent2_marker, = self.ax.plot([], [], 'mo', markersize=12, label=agent2_name, markeredgecolor='darkmagenta', markeredgewidth=2)
            if show_path:
                agent2_path, = self.ax.plot([], [], 'm-', alpha=0.3, linewidth=2)
        
        self.ax.set_xlim(-0.5, self.game.width - 0.5)
        self.ax.set_ylim(self.game.height - 0.5, -0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        self.ax.set_title(f'Pathfinding: {agent1_name} vs {agent2_name if agent2_positions else "Single Agent"}', 
                         fontsize=14, fontweight='bold')
        
        # Add step counter
        step_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                verticalalignment='top', fontsize=12, 
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        def animate(frame):
            """Animation function called for each frame."""
            # Update agent 1
            if frame < len(agent1_positions):
                r1, c1 = agent1_positions[frame]
                agent1_marker.set_data([c1], [r1])
                if agent1_path and show_path:
                    path_cols = [pos[1] for pos in agent1_positions[:frame+1]]
                    path_rows = [pos[0] for pos in agent1_positions[:frame+1]]
                    agent1_path.set_data(path_cols, path_rows)
            
            # Update agent 2
            if agent2_positions and agent2_marker:
                if frame < len(agent2_positions):
                    r2, c2 = agent2_positions[frame]
                    agent2_marker.set_data([c2], [r2])
                    if agent2_path and show_path:
                        path_cols = [pos[1] for pos in agent2_positions[:frame+1]]
                        path_rows = [pos[0] for pos in agent2_positions[:frame+1]]
                        agent2_path.set_data(path_cols, path_rows)
            
            # Update step counter
            step_text.set_text(f'Step: {frame + 1}/{max_steps}')
            
            return [agent1_marker, agent1_path, agent2_marker, agent2_path, step_text]
        
        # Create animation
        anim = animation.FuncAnimation(self.fig, animate, frames=max_steps, 
                                      interval=interval, blit=True, repeat=True)
        
        # Save if path provided
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else '.', exist_ok=True)
            # Save as GIF (requires pillow or imageio)
            try:
                anim.save(self.save_path, writer='pillow', fps=5)
                print(f"Animation saved to {self.save_path}")
            except Exception as e:
                print(f"Could not save animation as GIF: {e}")
                print("Showing animation instead...")
        
        print("Visualizing episode... Close the plot window to continue.")
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def plot_final_paths(self, agent1_positions: List[Tuple[int, int]], 
                        agent2_positions: Optional[List[Tuple[int, int]]] = None,
                        agent1_name: str = "Agent 1",
                        agent2_name: str = "Agent 2",
                        save_path: Optional[str] = None):
        """
        Plot final paths (static plot, no animation).
        
        Args:
            agent1_positions: List of positions for agent 1
            agent2_positions: Optional list of positions for agent 2
            agent1_name: Name for agent 1
            agent2_name: Name for agent 2
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        
        # Plot map (white walls on black background)
        map_grid = self.game.map_grid
        # Invert: walls (1) = white, free (0) = black
        inverted_map = 1 - map_grid  # Now 0=wall (black), 1=free (white)
        ax.imshow(inverted_map, cmap='gray', alpha=0.8, origin='upper', vmin=0, vmax=1)
        
        # Mark start and goal
        start_r, start_c = self.game.start_pos
        goal_r, goal_c = self.game.goal_pos
        
        ax.plot(start_c, start_r, 'go', markersize=15, label='Start (A)', 
               markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(goal_c, goal_r, 'r*', markersize=20, label='Goal (B)', 
               markeredgecolor='darkred', markeredgewidth=2)
        
        # Plot agent 1 path
        cols1 = [pos[1] for pos in agent1_positions]
        rows1 = [pos[0] for pos in agent1_positions]
        ax.plot(cols1, rows1, 'b-', linewidth=2, alpha=0.6, label=agent1_name)
        ax.plot(cols1[-1], rows1[-1], 'bo', markersize=10, markeredgecolor='darkblue', markeredgewidth=2)
        
        # Plot agent 2 path
        if agent2_positions:
            cols2 = [pos[1] for pos in agent2_positions]
            rows2 = [pos[0] for pos in agent2_positions]
            ax.plot(cols2, rows2, 'm-', linewidth=2, alpha=0.6, label=agent2_name)
            ax.plot(cols2[-1], rows2[-1], 'mo', markersize=10, markeredgecolor='darkmagenta', markeredgewidth=2)
        
        ax.set_xlim(-0.5, self.game.width - 0.5)
        ax.set_ylim(self.game.height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'Final Paths: {agent1_name} vs {agent2_name if agent2_positions else "Single Agent"}', 
                    fontsize=14, fontweight='bold')
        
        # Add statistics
        stats_text = f'Agent 1: {len(agent1_positions)} steps'
        if agent2_positions:
            stats_text += f'\nAgent 2: {len(agent2_positions)} steps'
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
               verticalalignment='bottom', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
