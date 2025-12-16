from abc import ABC, abstractmethod
import numpy as np


class BaseGame(ABC):
    """Abstract base class for zero-sum games."""
    
    def __init__(self, payoff_matrix):
        """
        Initialize game with payoff matrix.
        
        Args:
            payoff_matrix: 2D array where payoff_matrix[i][j] = payoff for row player
                          when row plays action i and column plays action j
        """
        self.payoff_matrix = np.array(payoff_matrix)
        self.n_actions = self.payoff_matrix.shape[0]
    
    @abstractmethod
    def get_nash_equilibrium(self):
        """Return the Nash equilibrium strategy for the row player."""
        pass
    
    def get_payoff(self, action1, action2):
        """
        Get payoff for row player.
        
        Args:
            action1: Action of row player
            action2: Action of column player
            
        Returns:
            Payoff for row player (column player gets -payoff in zero-sum)
        """
        return self.payoff_matrix[action1, action2]
    
    def get_actions(self):
        """Return list of available actions."""
        return list(range(self.n_actions))
    
    def best_response(self, opponent_strategy):
        """
        Compute best response to opponent's mixed strategy.
        
        Args:
            opponent_strategy: Probability distribution over opponent's actions
            
        Returns:
            Best response action (pure strategy)
        """
        opponent_strategy = np.array(opponent_strategy)
        expected_payoffs = self.payoff_matrix @ opponent_strategy
        return np.argmax(expected_payoffs)

