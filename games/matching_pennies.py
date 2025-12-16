import numpy as np
from .base_game import BaseGame


class MatchingPennies(BaseGame):
    """Matching Pennies: 2x2 zero-sum game."""
    
    def __init__(self):
        # Payoff matrix for row player
        # H=0, T=1
        payoff_matrix = [
            [1, -1],   # H
            [-1, 1]    # T
        ]
        super().__init__(payoff_matrix)
    
    def get_nash_equilibrium(self):
        """Nash equilibrium is uniform: (0.5, 0.5)"""
        return np.array([0.5, 0.5])
    
    def get_action_name(self, action):
        """Convert action index to name."""
        return ['H', 'T'][action]

