import numpy as np
from .base_game import BaseGame

class GridGame(BaseGame):
    """
    Grid Game (Hunter vs Prey) on a 3x3 board.
    
    State: (hunter_row, hunter_col, prey_row, prey_col) -> Flattened to 0..80
    Actions: 0:Up, 1:Right, 2:Down, 3:Left, 4:Stay
    
    Payoff:
    - If Hunter catches Prey (same cell): +10 for Hunter, -10 for Prey. Game Reset.
    - Else: -1 for Hunter (energy cost), +1 for Prey (survival bonus).
    """
    
    def __init__(self, size=3):
        # Dummy matrix for initialization (won't be used directly)
        super().__init__([[0]])
        
        self.size = size
        self.n_actions = 5 # Up, Right, Down, Left, Stay
        
        # Positions (row, col)
        self.hunter_pos = [0, 0]
        self.prey_pos = [size-1, size-1]
        
        # Directions mapping
        self.moves = {
            0: (-1, 0), # Up
            1: (0, 1),  # Right
            2: (1, 0),  # Down
            3: (0, -1), # Left
            4: (0, 0)   # Stay
        }
        
    def get_state(self):
        """
        Encode state as a single integer:
        h_row * (size^3) + h_col * (size^2) + p_row * size + p_col
        For 3x3, max index is 80.
        """
        h_r, h_c = self.hunter_pos
        p_r, p_c = self.prey_pos
        return h_r * (self.size**3) + h_c * (self.size**2) + p_r * self.size + p_c

    def _move(self, pos, action):
        """Calculate new position given action, checking bounds."""
        dr, dc = self.moves[action]
        new_r = max(0, min(self.size-1, pos[0] + dr))
        new_c = max(0, min(self.size-1, pos[1] + dc))
        return [new_r, new_c]

    def step(self, action1, action2):
        """
        Execute moves. P1 is Hunter, P2 is Prey.
        Simultaneous movement.
        """
        # Move Hunter
        self.hunter_pos = self._move(self.hunter_pos, action1)
        
        # Move Prey
        self.prey_pos = self._move(self.prey_pos, action2)
        
        # Check Collision
        if self.hunter_pos == self.prey_pos:
            reward = 10 # Capture
            # Reset positions
            self.hunter_pos = [0, 0]
            self.prey_pos = [self.size-1, self.size-1]
        else:
            reward = -1 # Hunger / Survival
            
        return reward, -reward # Zero-sum

    def get_nash_equilibrium(self):
        # Not applicable for Grid World in this framework
        return np.ones(self.n_actions) / self.n_actions
    
    def get_action_name(self, action):
        return ["Up", "Right", "Down", "Left", "Stay"][action]
