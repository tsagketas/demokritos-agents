import numpy as np
from .base_agent import BaseAgent

class StochasticQLearningAgent(BaseAgent):
    """
    Q-Learning agent for Stochastic Games (State-aware).
    """
    
    def __init__(self, n_actions, n_states=81, learning_rate=0.1, epsilon=0.1, 
                 lr_decay=0.99995, epsilon_decay=0.99995, name=None):
        super().__init__(n_actions, name)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Map state -> array of action values
        # Initialize lazily or fixed size
        self.n_states = n_states
        self.Q = {s: np.zeros(n_actions) for s in range(n_states)}
        self.last_state = 0
        self.player_id = 0 # Default
    
    def set_player_id(self, player_id):
        self.player_id = player_id

    def act(self, game=None):
        # Check current state
        if hasattr(game, 'get_state'):
            state = game.get_state()
        else:
            state = 0
        
        self.last_state = state
        
        # Epsilon-Greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Exploit
        q_values = self.Q[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    def update(self, action, reward, opponent_action=None):
        state = self.last_state
        
        # Q-Learning Update
        # Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
        # Note: Standard Q-learning adds gamma * max Q(s', a'). 
        # For simplicity in this zero-sum context, we treat it like independent bandits per state 
        # or simplified RL. Adding gamma requires knowing next_state in update, which we don't pass.
        # Given the interface constraints, we stick to Bandit-like update per state.
        
        self.Q[state][action] += self.learning_rate * (reward - self.Q[state][action])
        
        # Decay
        self.epsilon *= self.epsilon_decay
        self.learning_rate *= self.lr_decay
        
        self.reward_history.append(reward)

    def reset(self):
        super().reset()
        self.Q = {s: np.zeros(self.n_actions) for s in range(self.n_states)}
