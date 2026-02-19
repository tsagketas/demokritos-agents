from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for game-playing agents."""
    
    def __init__(self, n_actions, name=None):
        """
        Initialize agent.
        
        Args:
            n_actions: Number of available actions
            name: Optional name for the agent
        """
        self.n_actions = n_actions
        self.name = name or self.__class__.__name__
        self.player_id = 0  # 0 for Row Player, 1 for Column Player
        self.action_history = []
        self.reward_history = []
    
    def set_player_id(self, player_id):
        """Set player role (0 for Row, 1 for Column)."""
        self.player_id = player_id

    @abstractmethod
    def act(self, game=None):
        """
        Choose an action.
        
        Args:
            game: Optional game object (for context-dependent actions)
            
        Returns:
            Action index (0 to n_actions-1)
        """
        pass
    
    @abstractmethod
    def update(self, action, reward, opponent_action=None):
        """
        Update agent based on experience.
        
        Args:
            action: Action the agent took
            reward: Reward received
            opponent_action: Optional opponent's action
        """
        pass
    
    def get_strategy(self):
        """
        Get current mixed strategy (probability distribution over actions).
        
        Returns:
            Array of probabilities, shape (n_actions,)
        """
        if len(self.action_history) == 0:
            return np.ones(self.n_actions) / self.n_actions
        
        strategy = np.zeros(self.n_actions)
        for action in self.action_history:
            strategy[action] += 1
        return strategy / len(self.action_history)
    
    def reset(self):
        """Reset agent state."""
        self.action_history = []
        self.reward_history = []
    
    def get_cumulative_reward(self):
        """Get total cumulative reward."""
        return sum(self.reward_history) if self.reward_history else 0.0

