import numpy as np
from agents.base_agent import BaseAgent


class MinimaxAgent(BaseAgent):
    """
    Minimax agent for optimal play in turn-based, perfect information games.
    """
    
    def __init__(self, n_actions, depth=3, evaluation_function=None, name=None):
        """
        Initialize Minimax agent.
        
        Args:
            n_actions: Number of available actions
            depth: Search depth for Minimax
            evaluation_function: Function to evaluate non-terminal states
            name: Optional name for the agent
        """
        super().__init__(n_actions, name)
        self.depth = depth
        self.evaluation_function = evaluation_function
    
    def act(self, game):
        """
        Choose the best action using Minimax.
        
        Args:
            game: The game object to analyze
            
        Returns:
            Best action index
        """
        best_val = -float('inf')
        best_action = 0
        
        for action in range(self.n_actions):
            sim_game = game.clone()
            _, done, next_player = sim_game.step(action)
            
            if done:
                val = self._evaluate(sim_game)
            else:
                is_maximizing = (next_player == self.player_id)
                val = self._minimax(sim_game, self.depth - 1, is_maximizing)
            
            if val > best_val:
                best_val = val
                best_action = action
            
        return best_action
    
    def _minimax(self, game, depth, is_maximizing):
        """
        Recursive Minimax function.
        """
        if depth == 0:
            return self._evaluate(game)
            
        if is_maximizing:
            max_eval = -float('inf')
            for action in range(self.n_actions):
                sim_game = game.clone()
                _, done, next_player = sim_game.step(action)
                
                if done:
                    val = self._evaluate(sim_game)
                else:
                    val = self._minimax(sim_game, depth - 1, next_player == self.player_id)
                
                max_eval = max(max_eval, val)
            return max_eval
        else:
            min_eval = float('inf')
            for action in range(self.n_actions):
                sim_game = game.clone()
                _, done, next_player = sim_game.step(action)
                
                if done:
                    val = self._evaluate(sim_game)
                else:
                    val = self._minimax(sim_game, depth - 1, next_player == self.player_id)
                
                min_eval = min(min_eval, val)
            return min_eval
            
    def _evaluate(self, game):
        """
        Evaluate a game state using the provided evaluation function or a default.
        """
        if self.evaluation_function:
            return self.evaluation_function(game, self.player_id)
        
        # Default: if game is done, return a large value if we won
        # This requires the game to have a way to check rewards or winner
        # For now, we'll assume the game object can be queried for state value
        if hasattr(game, 'get_state_value'):
            return game.get_state_value(self.player_id)
            
        return 0.0

    def update(self, action, reward, opponent_action=None):
        """
        Minimax doesn't learn from experience, but we track history.
        """
        self.action_history.append(action)
        self.reward_history.append(reward)
