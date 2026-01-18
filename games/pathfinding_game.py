"""
Pathfinding Game: Gridworld/Maze where agents navigate from point A to point B.
Agents can move in 4 directions (up, down, left, right) and get rewards.
"""

import numpy as np
from typing import Tuple, List, Optional


class PathfindingGame:
    """
    Pathfinding game on a grid map.
    
    The game has:
    - A grid map (0 = free, 1 = wall/obstacle)
    - Start position (point A)
    - Goal position (point B)
    - Agents can move: UP, DOWN, LEFT, RIGHT (4 actions)
    - Rewards: +10 for reaching goal, -1 per step, -5 for hitting wall
    """
    
    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    # Direction vectors: (dy, dx)
    DIRECTIONS = [
        (-1, 0),  # UP
        (1, 0),   # DOWN
        (0, -1),  # LEFT
        (0, 1)    # RIGHT
    ]
    
    def __init__(self, map_grid: np.ndarray, start_pos: Tuple[int, int], 
                 goal_pos: Tuple[int, int], reward_goal: float = 10.0, 
                 reward_step: float = -0.1, reward_wall: float = -1.0):
        """
        Initialize pathfinding game.
        
        Args:
            map_grid: 2D numpy array, 0=free, 1=wall
            start_pos: (row, col) starting position
            goal_pos: (row, col) goal position
            reward_goal: Reward for reaching goal
            reward_step: Reward per step (typically negative)
            reward_wall: Reward for hitting wall
        """
        self.map_grid = np.array(map_grid, dtype=int)
        self.height, self.width = self.map_grid.shape
        self.start_pos = tuple(start_pos)
        self.goal_pos = tuple(goal_pos)
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_wall = reward_wall
        
        self.n_actions = 4  # UP, DOWN, LEFT, RIGHT
        
        # Validate positions
        self._validate_positions()
    
    def _validate_positions(self):
        """Validate that start and goal positions are valid."""
        sr, sc = self.start_pos
        gr, gc = self.goal_pos
        
        if not (0 <= sr < self.height and 0 <= sc < self.width):
            raise ValueError(f"Start position {self.start_pos} is out of bounds")
        if not (0 <= gr < self.height and 0 <= gc < self.width):
            raise ValueError(f"Goal position {self.goal_pos} is out of bounds")
        if self.map_grid[sr, sc] == 1:
            raise ValueError(f"Start position {self.start_pos} is on a wall")
        if self.map_grid[gr, gc] == 1:
            raise ValueError(f"Goal position {self.goal_pos} is on a wall")
        if self.start_pos == self.goal_pos:
            raise ValueError("Start and goal positions cannot be the same")
    
    def get_next_position(self, current_pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Get next position after taking action.
        
        Args:
            current_pos: (row, col) current position
            action: Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            (row, col) next position
        """
        row, col = current_pos
        dy, dx = self.DIRECTIONS[action]
        next_row = row + dy
        next_col = col + dx
        
        # Keep within bounds
        next_row = max(0, min(next_row, self.height - 1))
        next_col = max(0, min(next_col, self.width - 1))
        
        return (next_row, next_col)
    
    def is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall."""
        row, col = pos
        return self.map_grid[row, col] == 1
    
    def is_valid_move(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (not wall, within bounds).
        
        Args:
            pos: (row, col) position
            
        Returns:
            True if valid, False otherwise
        """
        row, col = pos
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return not self.is_wall(pos)
    
    def is_goal(self, pos: Tuple[int, int]) -> bool:
        """Check if position is the goal."""
        return pos == self.goal_pos
    
    def get_reward(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int], 
                   action: int) -> float:
        """
        Get reward for moving from old_pos to new_pos.
        
        Args:
            old_pos: Previous position
            new_pos: New position after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Reached goal
        if self.is_goal(new_pos):
            return self.reward_goal
        
        # Hit wall (stayed in same position or moved to wall)
        if new_pos == old_pos or self.is_wall(new_pos):
            return self.reward_wall
        
        # Normal step
        return self.reward_step
    
    def move(self, current_pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Move agent and get new position, reward, and done flag.
        
        Args:
            current_pos: Current (row, col) position
            action: Action index (0-3)
            
        Returns:
            (new_pos, reward, done) tuple
        """
        new_pos = self.get_next_position(current_pos, action)
        
        # If hitting wall, don't move
        if self.is_wall(new_pos):
            new_pos = current_pos
        
        reward = self.get_reward(current_pos, new_pos, action)
        done = self.is_goal(new_pos)
        
        return new_pos, reward, done
    
    def get_state_index(self, pos: Tuple[int, int]) -> int:
        """
        Convert position to state index (for tabular RL).
        
        Args:
            pos: (row, col) position
            
        Returns:
            State index (0 to height*width - 1)
        """
        row, col = pos
        return row * self.width + col
    
    def get_position_from_index(self, state_index: int) -> Tuple[int, int]:
        """Convert state index to position."""
        row = state_index // self.width
        col = state_index % self.width
        return (row, col)
    
    def get_optimal_path_length(self) -> Optional[int]:
        """
        Compute optimal path length using BFS (for comparison).
        
        Returns:
            Length of shortest path, or None if no path exists
        """
        from collections import deque
        
        queue = deque([(self.start_pos, 0)])
        visited = {self.start_pos}
        
        while queue:
            pos, steps = queue.popleft()
            
            if self.is_goal(pos):
                return steps
            
            for action in range(self.n_actions):
                next_pos = self.get_next_position(pos, action)
                if self.is_valid_move(next_pos) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, steps + 1))
        
        return None  # No path found
    
    @staticmethod
    def create_simple_maze(size: int = 10) -> 'PathfindingGame':
        """
        Create a simple maze for testing.
        
        Args:
            size: Grid size (size x size)
            
        Returns:
            PathfindingGame instance
        """
        # Create empty grid
        map_grid = np.zeros((size, size), dtype=int)
        
        # Add some walls (create a simple path)
        # Leave a clear path from top-left to bottom-right
        if size >= 10:
            # Add some obstacles but leave path open
            map_grid[3, 2:8] = 1  # Horizontal wall
            map_grid[6, 3:7] = 1  # Another horizontal wall
        
        start = (0, 0)
        goal = (size - 1, size - 1)
        
        return PathfindingGame(map_grid, start, goal)
    
    @staticmethod
    def create_empty_room(width: int = 10, height: int = 10) -> 'PathfindingGame':
        """
        Create empty room (no obstacles).
        
        Args:
            width: Room width
            height: Room height
            
        Returns:
            PathfindingGame instance
        """
        map_grid = np.zeros((height, width), dtype=int)
        start = (0, 0)
        goal = (height - 1, width - 1)
        
        return PathfindingGame(map_grid, start, goal)
    
    @staticmethod
    def create_hard_maze(size: int = 15) -> 'PathfindingGame':
        """
        Create harder maze with more obstacles.
        
        Args:
            size: Grid size
            
        Returns:
            PathfindingGame instance
        """
        map_grid = np.zeros((size, size), dtype=int)
        
        # Create a more complex maze
        # Walls with gaps
        map_grid[2, 1:size-1] = 1
        map_grid[2, 5] = 0  # Gap
        
        map_grid[5, 1:size-1] = 1
        map_grid[5, 10] = 0  # Gap
        
        map_grid[8, 1:size-1] = 1
        map_grid[8, 3] = 0  # Gap
        
        map_grid[11, 1:size-1] = 1
        map_grid[11, 8] = 0  # Gap
        
        # Vertical walls
        map_grid[3:size-1, 3] = 1
        map_grid[5, 3] = 0  # Gap
        
        map_grid[1:8, 7] = 1
        map_grid[3, 7] = 0  # Gap
        
        start = (0, 0)
        goal = (size - 1, size - 1)
        
        return PathfindingGame(map_grid, start, goal)
    
    @staticmethod
    def create_from_athens_map(image_path: str = "athens_map.png",
                               start_pos: Tuple[int, int] = None,
                               goal_pos: Tuple[int, int] = None,
                               size: Tuple[int, int] = (60, 60)) -> 'PathfindingGame':
        """
        Create game from Athens map image.
        Black pixels = walls, white pixels = free space.
        
        Args:
            image_path: Path to Athens map image (if doesn't exist, generates one)
            start_pos: Start position (default: top-left free space)
            goal_pos: Goal position (default: bottom-right free space)
            size: Size for generated map if image doesn't exist
            
        Returns:
            PathfindingGame instance
        """
        from .map_loader import MapLoader
        
        # Load map
        map_grid = MapLoader.create_athens_from_image(image_path, threshold=128)
        
        # Resize if needed
        if size and (map_grid.shape[0] != size[0] or map_grid.shape[1] != size[1]):
            map_grid = MapLoader.resize_map(map_grid, size)
        
        # Find start and goal positions (free spaces)
        free_spaces = np.where(map_grid == 0)
        free_positions = list(zip(free_spaces[0], free_spaces[1]))
        
        if start_pos is None:
            # Top-left free space
            start_pos = free_positions[0] if free_positions else (1, 1)
        
        if goal_pos is None:
            # Bottom-right free space
            goal_pos = free_positions[-1] if free_positions else (map_grid.shape[0]-2, map_grid.shape[1]-2)
        
        return PathfindingGame(map_grid, start_pos, goal_pos)
