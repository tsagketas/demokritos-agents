import numpy as np
from .base_game import BaseGame


def _manhattan(hunter_pos, prey_pos):
    return abs(hunter_pos[0] - prey_pos[0]) + abs(hunter_pos[1] - prey_pos[1])


class GridGame(BaseGame):
    """
    Turn-based Grid Game (Hunter vs Prey) on a 3x3 board.
    
    Turn order: Hunter (player 0) → Prey (player 1) → Hunter → ...
    Perfect information. One player moves per step.
    
    State: (hunter_row, hunter_col, prey_row, prey_col) → flattened 0..80
    Actions: 0:Up, 1:Right, 2:Down, 3:Left, 4:Stay
    
    Rewards:
    - Capture (Hunter lands on Prey): +10 Hunter, -10 Prey
    - Timeout: -10 Hunter, +10 Prey
    - Non-terminal: distance-based shaping (delta_dist - 0.1 for Hunter, -delta_dist + 0.1 for Prey)
    """
    
    def __init__(self, size=3, max_steps=20, first_player=0):
        super().__init__([[0]])
        self.size = size
        self.max_steps = max_steps
        self.first_player = first_player  # 0 = Hunter first, 1 = Prey first
        self.steps = 0
        self.n_actions = 5
        self.hunter_pos = [0, 0]
        self.prey_pos = [size - 1, size - 1]
        self.current_player = 0  # 0 = Hunter, 1 = Prey
        self.moves = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 0),   # Stay
        }

    def get_state(self):
        """State index 0..80 for 3x3: h_r*27 + h_c*9 + p_r*3 + p_c."""
        h_r, h_c = self.hunter_pos
        p_r, p_c = self.prey_pos
        return h_r * (self.size ** 3) + h_c * (self.size ** 2) + p_r * self.size + p_c

    def get_current_player(self):
        """Who moves next: 0 = Hunter, 1 = Prey."""
        return self.current_player

    def _move(self, pos, action):
        dr, dc = self.moves[action]
        new_r = max(0, min(self.size - 1, pos[0] + dr))
        new_c = max(0, min(self.size - 1, pos[1] + dc))
        return [new_r, new_c]

    def reset(self):
        self.hunter_pos = [0, 0]
        self.prey_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        self.current_player = self.first_player
        return self.get_state()

    def step(self, action):
        """
        One player moves. Returns (reward, done, next_player).
        reward is for the player who just moved.
        """
        self.steps += 1
        old_dist = _manhattan(self.hunter_pos, self.prey_pos)
        reward = 0.0
        done = False

        if self.current_player == 0:
            # Hunter moves
            self.hunter_pos = self._move(self.hunter_pos, action)
            new_dist = _manhattan(self.hunter_pos, self.prey_pos)
            if self.hunter_pos == self.prey_pos:
                reward = 10
                done = True
                next_player = 0  # irrelevant
            elif self.steps >= self.max_steps:
                reward = -10
                done = True
                next_player = 0
            else:
                delta = old_dist - new_dist
                reward = delta - 0.1
                next_player = 1
        else:
            # Prey moves
            self.prey_pos = self._move(self.prey_pos, action)
            new_dist = _manhattan(self.hunter_pos, self.prey_pos)
            if self.steps >= self.max_steps:
                reward = 10
                done = True
                next_player = 0
            else:
                delta = old_dist - new_dist
                reward = -delta + 0.1
                next_player = 0

        self.current_player = next_player
        return reward, done, next_player

    def clone(self):
        """Copy state for simulation (e.g. Minimax)."""
        c = GridGame(size=self.size, max_steps=self.max_steps, first_player=self.first_player)
        c.steps = self.steps
        c.hunter_pos = list(self.hunter_pos)
        c.prey_pos = list(self.prey_pos)
        c.current_player = self.current_player
        return c

    def get_state_value(self, player_id):
        """
        Value of current state for player_id (for Minimax evaluation).
        Terminal: Capture +10/-10, Timeout -10/+10. Non-terminal: distance heuristic.
        """
        dist = _manhattan(self.hunter_pos, self.prey_pos)
        if self.hunter_pos == self.prey_pos:
            return 10.0 if player_id == 0 else -10.0
        if self.steps >= self.max_steps:
            return -10.0 if player_id == 0 else 10.0
        # Non-terminal: Hunter prefers lower distance, Prey higher
        if player_id == 0:
            return -dist  # Hunter maximizes → closer is better
        return dist     # Prey maximizes → farther is better

    def get_nash_equilibrium(self):
        return np.ones(self.n_actions) / self.n_actions

    def get_action_name(self, action):
        return ["Up", "Right", "Down", "Left", "Stay"][action]
