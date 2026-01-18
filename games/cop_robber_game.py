
import numpy as np
from typing import Tuple, List, Optional
from .pathfinding_game import PathfindingGame

class CopRobberGame(PathfindingGame):
    """
    Stochastic Zero-Sum Cop and Robber Game.
    
    Dynamics:
    - Cop tries to catch Robber (same position).
    - Robber tries to evade Cop.
    - Stochastic: 10% chance (configurable) that a move 'slips' to a random neighbor.
    
    Zero-Sum Rewards:
    - If caught: Cop +100, Robber -100
    - Per step: Cop -1 (cost of time), Robber +1 (reward for survival)
    """
    
    def __init__(self, map_grid: np.ndarray, cop_start: Tuple[int, int], 
                 robber_start: Tuple[int, int], max_steps: int = 100,
                 slip_prob: float = 0.1):
        """
        Initialize Cop & Robber game.
        """
        # We don't use 'start' and 'goal' in the same way, but we init parent
        super().__init__(map_grid, cop_start, robber_start)
        
        self.cop_pos = cop_start
        self.robber_pos = robber_start
        self.max_steps = max_steps
        self.slip_prob = slip_prob
        self.current_step = 0
        
    def reset(self):
        """Reset game state."""
        self.cop_pos = self.start_pos  # Stored in parent as start
        self.robber_pos = self.goal_pos # Stored in parent as goal
        self.current_step = 0
        return self.get_state()
        
    def get_state(self):
        """
        Return the full state: (cop_position, robber_position)
        """
        return (self.cop_pos, self.robber_pos)
    
    def step(self, cop_action: int, robber_action: int) -> Tuple[Tuple, Tuple, bool, str]:
        """
        Execute one step for both agents simultaneously.
        
        Args:
            cop_action: 0-3
            robber_action: 0-3
            
        Returns:
            next_state: (cop_pos, robber_pos)
            rewards: (cop_reward, robber_reward)
            done: bool
            info: str (reason for end)
        """
        self.current_step += 1
        
        # 1. Determine Intended Moves
        cop_next = self._get_stochastic_move(self.cop_pos, cop_action)
        robber_next = self._get_stochastic_move(self.robber_pos, robber_action)
        
        # 2. Resolve Collisions/Walls
        # If move into wall, stay put
        if self.is_wall(cop_next):
            cop_next = self.cop_pos
        
        if self.is_wall(robber_next):
            robber_next = self.robber_pos
            
        # 3. Update Positions
        self.cop_pos = cop_next
        self.robber_pos = robber_next
        
        # 4. Check Capture
        # Capture happens if they are on same square OR they swapped squares (passed through)
        caught = (self.cop_pos == self.robber_pos)
        
        # 5. Calculate Zero-Sum Rewards
        cop_reward = -1  # Penalty for time passing
        robber_reward = 1 # Reward for surviving
        
        done = False
        info = "step"
        
        if caught:
            cop_reward += 100
            robber_reward -= 100
            done = True
            info = "caught"
        elif self.current_step >= self.max_steps:
            done = True
            info = "timeout"
            
        return (self.cop_pos, self.robber_pos), (cop_reward, robber_reward), done, info

    def _get_stochastic_move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Apply action with slip probability."""
        if np.random.random() < self.slip_prob:
            # SLIP! Random valid move (including staying still or intended)
            # Simple stochasticity: Random direction 0-3
            random_action = np.random.randint(4)
            return self.get_next_position(pos, random_action)
        else:
            return self.get_next_position(pos, action)
            
    def get_state_index_combined(self) -> int:
        """
        Returns a unique integer index for the combined state (cop, robber).
        Warning: For large maps, this is huge (N*N * N*N).
        For 50x50 map: 2500 * 2500 = 6,250,000 states.
        """
        cop_idx = self.cop_pos[0] * self.width + self.cop_pos[1]
        rob_idx = self.robber_pos[0] * self.width + self.robber_pos[1]
        return cop_idx * (self.width * self.height) + rob_idx

