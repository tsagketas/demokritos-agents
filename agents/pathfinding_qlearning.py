"""
Q-Learning agent adapted for pathfinding games.
Uses state-based Q-values (state = position on grid).
"""

import numpy as np
from .base_agent import BaseAgent
from typing import Optional, Tuple


class PathfindingQLearningAgent(BaseAgent):
    """
    Q-Learning agent for pathfinding.
    
    State space: All grid positions (state_index = row * width + col)
    Action space: 4 actions (UP, DOWN, LEFT, RIGHT)
    Q-values: Q(state, action) table
    """
    
    def __init__(self, n_states: int, n_actions: int = 4, learning_rate: float = 0.1,
                 epsilon: float = 0.2, discount: float = 0.99, name: Optional[str] = None,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        Initialize Pathfinding Q-Learning agent.
        
        Args:
            n_states: Number of states (height * width)
            n_actions: Number of actions (4: up, down, left, right)
            learning_rate: Learning rate (alpha)
            epsilon: Exploration rate (epsilon-greedy)
            discount: Discount factor (gamma)
            name: Agent name
        """
        super().__init__(n_actions, name or "PathfindingQL")
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount = discount
        
        # Q-table: Q[state, action] - NOT reset between episodes (keeps learning!)
        self.Q = np.zeros((n_states, n_actions))
        
        # Current state tracking
        self.current_state = None
        self.position_history = []
    
    def act(self, state: int, game=None) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state index
            game: Game object (optional, for future use)
            
        Returns:
            Action index (0-3)
        """
        self.current_state = state
        
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action according to Q-values
            q_values = self.Q[state, :]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state: int, action: int, reward: float, next_state: Optional[int] = None, 
               done: bool = False, opponent_action=None):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if terminal)
            done: Whether episode is done
            opponent_action: Not used (kept for interface compatibility)
        """
        # Q-learning update
        if done or next_state is None:
            # Terminal state or episode end
            target = reward
        else:
            # Normal update: r + γ*max(Q(s',a'))
            target = reward + self.discount * np.max(self.Q[next_state, :])
        
        # Q(s,a) ← Q(s,a) + α[target - Q(s,a)]
        self.Q[state, action] += self.learning_rate * (target - self.Q[state, action])
        
        # Track history
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def reset(self):
        """
        Reset agent for new episode.
        NOTE: Q-table is NOT reset - agent keeps learning across episodes!
        Only episode-specific state is reset.
        """
        super().reset()
        self.current_state = None
        self.position_history = []
        
        # Decay epsilon (explore less over time)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self):
        """Get current Q-table."""
        return self.Q.copy()
    
    def get_strategy(self):
        """
        Get current strategy (softmax over Q-values for current state).
        If no current state, return uniform distribution.
        """
        if self.current_state is None:
            return np.ones(self.n_actions) / self.n_actions
        
        q_values = self.Q[self.current_state, :]
        # Softmax with temperature
        exp_q = np.exp(q_values - np.max(q_values))
        return exp_q / np.sum(exp_q)

    def save(self, filepath: str):
        """Save Q-table to file."""
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        np.save(filepath, self.Q)
        
    def load(self, filepath: str):
        """Load Q-table from file."""
        self.Q = np.load(filepath)
