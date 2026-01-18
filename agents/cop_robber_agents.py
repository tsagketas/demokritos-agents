
import numpy as np
from .base_agent import BaseAgent
from collections import deque

class GreedyCopAgent(BaseAgent):
    """
    Greedy Cop: Always moves one step closer to the Robber (BFS shortest path).
    """
    def __init__(self, name="Cop"):
        super().__init__(4, name)
        
    def act(self, game_state, game=None):
        """
        game_state: (cop_pos, robber_pos)
        """
        if game is None:
            raise ValueError("Game object required for BFS")
            
        cop_pos, robber_pos = game_state
        
        # Use BFS to find shortest path to robber
        path = self._bfs_path(game, cop_pos, robber_pos)
        
        if not path or len(path) < 2:
            return np.random.randint(4) # Stuck or already there
            
        next_pos = path[1]
        
        # Determine action to get there
        for action in range(4):
            if game.get_next_position(cop_pos, action) == next_pos:
                return action
        
        return 0 

    def update(self, action, reward, opponent_action=None):
        """Greedy agent doesn't learn, but tracks history."""
        self.action_history.append(action)
        self.reward_history.append(reward)

    def _bfs_path(self, game, start, end):
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            curr, path = queue.popleft()
            if curr == end:
                return path
            
            for action in range(4):
                neighbor = game.get_next_position(curr, action)
                if game.is_valid_move(neighbor) and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

class GreedyRobberAgent(BaseAgent):
    """
    Greedy Robber: Moves to the neighbor that maximizes distance to Cop.
    """
    def __init__(self, name="Robber"):
        super().__init__(4, name)
        
    def act(self, game_state, game=None):
        if game is None:
            return np.random.randint(4)
            
        cop_pos, robber_pos = game_state
        
        best_action = np.random.randint(4)
        max_dist = -1
        
        # Check all valid moves
        valid_moves = []
        for action in range(4):
            next_pos = game.get_next_position(robber_pos, action)
            if game.is_valid_move(next_pos):
                # Calculate Manhattan distance to cop
                dist = abs(next_pos[0] - cop_pos[0]) + abs(next_pos[1] - cop_pos[1])
                
                if dist > max_dist:
                    max_dist = dist
                    best_action = action
                    valid_moves = [action]
                elif dist == max_dist:
                    valid_moves.append(action)
        
        if valid_moves:
            return np.random.choice(valid_moves)
            
        return best_action

    def update(self, action, reward, opponent_action=None):
        """Greedy agent doesn't learn, but tracks history."""
        self.action_history.append(action)
        self.reward_history.append(reward)
