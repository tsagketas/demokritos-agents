import numpy as np
from .base_agent import BaseAgent

class StochasticQLearningAgent(BaseAgent):
    """
    Q-Learning agent for Stochastic Games (State-aware).
    """
    
    def __init__(self, n_actions, n_states=81, learning_rate=0.1, epsilon=0.1, 
                 lr_decay=0.99995, epsilon_decay=0.99995, discount_factor=0.95, name=None):
        super().__init__(n_actions, name)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.epsilon_decay = epsilon_decay
        self.gamma = discount_factor
        
        # Q-table: Map state -> array of action values
        # Initialize lazily or fixed size
        self.n_states = n_states
        self.Q = {s: np.zeros(n_actions) for s in range(n_states)}
        self.last_state = 0
        self.player_id = 0 # Default
    
    def set_player_id(self, player_id):
        self.player_id = player_id

    def act(self, game=None, state=None):
        # Check current state
        if state is None:
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
    
    def update(self, action, reward, next_state=None, done=False):
        state = self.last_state
        
        # Q-Learning Update
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max Q(s',a') - Q(s,a))
        
        if next_state is None:
            # Fallback for stateless/unknown next state
            target = reward
        else:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state][action] += self.learning_rate * (target - self.Q[state][action])
        
        # Decay
        self.epsilon *= self.epsilon_decay
        self.learning_rate *= self.lr_decay
        
        self.reward_history.append(reward)

    def reset(self):
        super().reset()
        self.Q = {s: np.zeros(self.n_actions) for s in range(self.n_states)}
