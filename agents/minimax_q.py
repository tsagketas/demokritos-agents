
import numpy as np
from scipy.optimize import linprog
from .base_agent import BaseAgent

class MinimaxQAgent(BaseAgent):
    """
    Minimax-Q Agent (Littman, 1994).
    Learns to play optimal zero-sum strategies in Markov Games.
    
    Maintains Q(s, a, o) table:
    - s: state index
    - a: my action
    - o: opponent action
    """
    
    def __init__(self, n_states, n_actions=4, learning_rate=1.0, 
                 discount=0.9, decay=0.99995, min_lr=0.001, name="MinimaxQ"):
        super().__init__(n_actions, name)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount
        self.decay = decay
        self.min_lr = min_lr
        
        # Q-Table: [State, MyAction, OppAction]
        # Initialize optimistically or with zeros
        self.Q = np.zeros((n_states, n_actions, n_actions))
        
        # Value of each state (V(s))
        self.V = np.zeros(n_states)
        
        # Current policy for each state (pi(s))
        self.policy = np.ones((n_states, n_actions)) / n_actions
        
    def act(self, state_idx, game=None):
        """
        Choose action based on current policy for state.
        Uses epsilon-greedy for exploration (though Minimax-Q is essentially greedy w.r.t value).
        """
        # Exploration is handled by the policy probabilities or explicit epsilon?
        # Standard Minimax-Q executes the mixed strategy derived from the LP.
        # But we need some exploration to visit all (s, a, o).
        # We'll use a simple epsilon-mixture: (1-e)*Optimal + e*Random
        
        # Calculate current exploration rate (decaying alpha acts as exploration proxy in some variants,
        # but explicit epsilon is safer).
        epsilon = max(0.01, self.alpha) # Heuristic: explore more when learning rate is high
        
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Sample from the mixed strategy pi(s)
            probs = self.policy[state_idx]
            # Normalize just in case
            probs = probs / np.sum(probs)
            return np.random.choice(self.n_actions, p=probs)
            
    def update(self, state, action, reward, next_state, done, opponent_action):
        """
        Update Q(s, a, o) and V(s).
        
        Args:
            state: Current state index
            action: My action index
            reward: Reward received
            next_state: Next state index
            done: Bool
            opponent_action: Opponent's action index (REQUIRED)
        """
        if opponent_action is None:
            raise ValueError("Minimax-Q requires opponent_action for update")
            
        # 1. Update Q(s, a, o)
        # Target = Reward + Gamma * V(next_state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.V[next_state]
            
        current_q = self.Q[state, action, opponent_action]
        self.Q[state, action, opponent_action] = current_q + self.alpha * (target - current_q)
        
        # 2. Update Policy pi(s) and Value V(s) by solving LP
        self._solve_linear_program(state)
        
        # 3. Decay learning rate
        self.alpha = max(self.min_lr, self.alpha * self.decay)
        
    def _solve_linear_program(self, state):
        """
        Solve Maximin LP for the given state.
        Max v
        s.t. sum(pi_a * Q(s, a, o)) >= v  for all o (opponent actions)
             sum(pi_a) = 1
             pi_a >= 0
        """
        # Scipy linprog minimizes c^T x.
        # We want to Maximize v => Minimize -v.
        # Variables x = [v, pi_0, pi_1, pi_2, pi_3] (size 1 + n_actions)
        
        c = [-1.0] + [0.0] * self.n_actions # Minimize -v (Maximize v)
        
        # Constraints:
        # sum(pi_a * Q(s, a, o)) >= v  =>  v - sum(...) <= 0
        # Coefficient for v is 1.
        # Coefficients for pi_a are -Q(s, a, o).
        
        A_ub = []
        b_ub = []
        
        # Q matrix for this state: Rows=MyAction, Cols=OppAction
        # We want constraints for each column (opp action)
        Q_s = self.Q[state] # shape (4, 4)
        
        for o in range(self.n_actions):
            # Constraint for opponent action o:
            # 1*v + sum_a( -Q[s, a, o] * pi_a ) <= 0
            row = [1.0]
            for a in range(self.n_actions):
                row.append(-Q_s[a, o])
            A_ub.append(row)
            b_ub.append(0.0)
            
        # Equality constraint: sum(pi) = 1
        # 0*v + 1*pi_0 + ... = 1
        A_eq = [[0.0] + [1.0] * self.n_actions]
        b_eq = [1.0]
        
        # Bounds: v is free (None, None), pi_a >= 0 (0, None)
        bounds = [(None, None)] + [(0.0, None)] * self.n_actions
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                # Extract results
                v = res.x[0]
                pi = res.x[1:]
                
                # Update Value and Policy
                self.V[state] = v
                self.policy[state] = pi
            else:
                # Fallback if LP fails (rare): keep old policy or uniform
                pass
        except Exception as e:
            # print(f"LP Error: {e}")
            pass

    def save(self, filepath: str):
        """Save Q-table to file."""
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        np.savez(filepath, Q=self.Q, V=self.V, policy=self.policy)
        
    def load(self, filepath: str):
        """Load Q-table from file."""
        data = np.load(filepath)
        self.Q = data['Q']
        self.V = data['V']
        self.policy = data['policy']
