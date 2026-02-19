import numpy as np
from .base_agent import BaseAgent


class FictitiousPlayAgent(BaseAgent):
    """Fictitious Play agent: learns opponent's strategy and plays best response."""
    
    def __init__(self, n_actions, name=None):
        """
        Initialize Fictitious Play agent.
        
        Args:
            n_actions: Number of available actions
            name: Optional name for the agent
        """
        super().__init__(n_actions, name)
        self.opponent_history = []
        self.action_counts = np.zeros(n_actions) # Count of each action played by opponent
        self.belief = np.ones(n_actions) / n_actions  # Uniform initial belief
    
    def act(self, game=None):
        """
        Choose action: best response to current belief about opponent.
        
        Args:
            game: Game object (required for best response calculation)
            
        Returns:
            Action index (best response to belief)
        """
        if game is None:
            raise ValueError("Game object required for Fictitious Play")
        
        return game.best_response(self.belief, self.player_id)
    
    def update(self, action, reward, opponent_action=None):
        """
        Update belief based on opponent's action.
        
        Args:
            action: Action the agent took (not used in FP)
            reward: Reward received (not used in FP)
            opponent_action: Opponent's action (required)
        """
        if opponent_action is None:
            raise ValueError("Opponent action required for Fictitious Play update")
        
        self.opponent_history.append(opponent_action)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Incremental update: O(1)
        self.action_counts[opponent_action] += 1
        total_actions = len(self.opponent_history)
        self.belief = self.action_counts / total_actions
    
    def _update_belief(self):
        """Deprecated: Internal method for belief update (now handled incrementally)."""
        pass
    
    def get_belief(self):
        """Get current belief about opponent's strategy."""
        return self.belief.copy()
    
    def reset(self):
        """Reset agent state."""
        super().reset()
        self.opponent_history = []
        self.action_counts = np.zeros(self.n_actions)
        self.belief = np.ones(self.n_actions) / self.n_actions

