import numpy as np
from .base_agent import BaseAgent

class StochasticFPAgent(BaseAgent):
    """
    Stochastic Fictitious Play Agent (FP-Q).
    
    Learns Q(s, a, o) like Minimax-Q, but updates Value based on 
    the EMPIRICAL DISTRIBUTION of the opponent, not the worst case.
    
    V(s) = max_a sum_o ( Belief(o|s) * Q(s, a, o) )
    """
    
    def __init__(self, n_states, n_actions=4, learning_rate=1.0, 
                 discount=0.9, decay=0.99995, min_lr=0.001, name="StochasticFP"):
        super().__init__(n_actions, name)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount
        self.decay = decay
        self.min_lr = min_lr
        
        # Q-Table: [State, MyAction, OppAction]
        self.Q = np.zeros((n_states, n_actions, n_actions))
        
        # Beliefs: Count of opponent actions in each state
        self.opponent_counts = np.ones((n_states, n_actions)) # Prior of 1
        
        # Value of each state (V(s))
        self.V = np.zeros(n_states)
        
    def act(self, state_idx, game=None):
        """
        Play Best Response to Current Belief.
        """
        # 1. Compute Belief pi(o|s)
        counts = self.opponent_counts[state_idx]
        belief = counts / np.sum(counts)
        
        # 2. Compute Expected Q for each action: E[Q(s,a)] = sum_o belief(o) * Q(s,a,o)
        expected_q = np.dot(self.Q[state_idx], belief)
        
        # 3. Choose Greedy Action (Best Response)
        # Add small epsilon for exploration
        epsilon = max(0.01, self.alpha)
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Random tie-breaking
            best_val = np.max(expected_q)
            best_actions = np.where(expected_q == best_val)[0]
            return np.random.choice(best_actions)
            
    def update(self, state, action, reward, next_state, done, opponent_action):
        """
        Update Beliefs, Q-values, and V(s).
        """
        if opponent_action is None:
            raise ValueError("FP requires opponent_action")
            
        # 1. Update Beliefs
        self.opponent_counts[state, opponent_action] += 1
        
        # 2. Update Q(s, a, o)
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.V[next_state]
            
        current_q = self.Q[state, action, opponent_action]
        self.Q[state, action, opponent_action] = current_q + self.alpha * (target - current_q)
        
        # 3. Update V(s) using new Beliefs and Q
        # V(s) = max_a sum_o pi(o|s) * Q(s, a, o)
        counts = self.opponent_counts[state]
        belief = counts / np.sum(counts)
        
        # Expected value for each of my actions
        expected_q_values = np.dot(self.Q[state], belief)
        
        # Value of state is the best I can do against this belief
        self.V[state] = np.max(expected_q_values)
        
        # 4. Decay learning rate
        self.alpha = max(self.min_lr, self.alpha * self.decay)
        
    def save(self, filepath: str):
        """Save Q-table to file."""
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        np.savez(filepath, Q=self.Q, V=self.V, counts=self.opponent_counts)
        
    def load(self, filepath: str):
        """Load Q-table from file."""
        data = np.load(filepath)
        self.Q = data['Q']
        self.V = data['V']
        self.opponent_counts = data['counts']