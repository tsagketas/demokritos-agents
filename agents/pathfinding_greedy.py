"""
Greedy/Fictitious Play-like agent for pathfinding.
Uses heuristic: tries to move towards goal (greedy best response).
Similar to FP in that it uses a "best response" strategy based on beliefs.
"""

import numpy as np
from .base_agent import BaseAgent
from typing import Tuple, Optional


class PathfindingGreedyAgent(BaseAgent):
    """
    Greedy agent that always moves towards the goal.
    Similar to FP in spirit: uses deterministic best response strategy.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize greedy pathfinding agent.
        
        Args:
            name: Agent name
        """
        super().__init__(n_actions=4, name=name or "PathfindingGreedy")
    
    def act(self, current_pos: Tuple[int, int], goal_pos: Tuple[int, int], 
            game=None) -> int:
        """
        Choose action that moves towards goal (greedy).
        Checks for walls and avoids them.
        
        Args:
            current_pos: Current (row, col) position
            goal_pos: Goal (row, col) position
            game: Game object (REQUIRED for wall checking)
            
        Returns:
            Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        if game is None:
            raise ValueError("Game object required for greedy agent to check walls")
        
        curr_row, curr_col = current_pos
        goal_row, goal_col = goal_pos
        
        # Calculate direction to goal
        dr = goal_row - curr_row  # Positive = need to go down
        dc = goal_col - curr_col  # Positive = need to go right
        
        # Try actions in order of preference (towards goal)
        # If a wall is hit, try next best option
        actions_to_try = []
        
        # Prioritize actions based on direction to goal
        if abs(dc) > abs(dr):
            # Prefer horizontal movement
            if dc > 0:
                actions_to_try = [3, 1, 0, 2]  # RIGHT, then vertical, then LEFT
            elif dc < 0:
                actions_to_try = [2, 1, 0, 3]  # LEFT, then vertical, then RIGHT
            else:
                actions_to_try = [1, 0, 3, 2]  # Vertical only
        else:
            # Prefer vertical movement
            if dr > 0:
                actions_to_try = [1, 3, 2, 0]  # DOWN, then horizontal, then UP
            elif dr < 0:
                actions_to_try = [0, 3, 2, 1]  # UP, then horizontal, then DOWN
            else:
                actions_to_try = [3, 2, 1, 0]  # Horizontal only
        
        # If at goal, choose randomly (shouldn't happen but safety)
        if dr == 0 and dc == 0:
            return np.random.randint(4)
        
        # Try actions in priority order, skip if wall
        for action in actions_to_try:
            next_pos = game.get_next_position(current_pos, action)
            if game.is_valid_move(next_pos):
                return action
        
        # If all moves blocked (shouldn't happen normally), choose random valid move
        for action in range(4):
            next_pos = game.get_next_position(current_pos, action)
            if game.is_valid_move(next_pos):
                return action
        
        # Last resort: return first action (shouldn't get here)
        return 0
    
    def update(self, action, reward, opponent_action=None):
        """
        Update agent (greedy doesn't learn, but tracks history).
        
        Args:
            action: Action taken
            reward: Reward received
            opponent_action: Not used
        """
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def get_strategy(self):
        """
        Greedy agent doesn't have a mixed strategy (always deterministic),
        but return uniform for compatibility.
        """
        return np.ones(self.n_actions) / self.n_actions
