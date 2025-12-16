import numpy as np
from .base_game import BaseGame


class RockPaperScissors(BaseGame):
    """Rock-Paper-Scissors: 3x3 zero-sum game."""
    
    def __init__(self):
        # Payoff matrix for row player
        # R=0, P=1, S=2
        payoff_matrix = [
            [0, -1, 1],   # Rock
            [1, 0, -1],   # Paper
            [-1, 1, 0]    # Scissors
        ]
        super().__init__(payoff_matrix)
    
    def get_nash_equilibrium(self):
        """Nash equilibrium is uniform: (1/3, 1/3, 1/3)"""
        return np.array([1/3, 1/3, 1/3])
    
    def get_action_name(self, action):
        """Convert action index to name."""
        return ['R', 'P', 'S'][action]

