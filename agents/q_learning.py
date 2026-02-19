import numpy as np
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Q-Learning agent: learns action values through exploration and exploitation."""
    
    def __init__(self, n_actions, learning_rate=0.1, epsilon=0.1, 
                 lr_decay=0.99995, epsilon_decay=0.99995, 
                 min_lr=0.001, min_epsilon=0.01, name=None):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_actions: Number of available actions
            learning_rate: Initial learning rate (alpha)
            epsilon: Initial exploration rate
            lr_decay: Multiplier for learning rate decay per update
            epsilon_decay: Multiplier for epsilon decay per update
            min_lr: Minimum learning rate
            min_epsilon: Minimum epsilon
            name: Optional name for the agent
        """
        super().__init__(n_actions, name)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.epsilon_decay = epsilon_decay
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.initial_lr = learning_rate
        self.initial_epsilon = epsilon
        self.Q = np.zeros(n_actions)  # Q-values for each action
    
    def act(self, game=None):
        """
        Choose action: epsilon-greedy policy.
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action according to Q-values
            max_q = np.max(self.Q)
            # Find all actions with max Q to break ties randomly
            best_actions = np.where(self.Q == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, action, reward, opponent_action=None):
        """
        Update Q-value and decay parameters.
        """
        # Q-learning update: Q(a) = Q(a) + alpha * (reward - Q(a))
        self.Q[action] += self.learning_rate * (reward - self.Q[action])
        
        # Decay parameters
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay)
        
        # Track history
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def get_q_values(self):
        """Get current Q-values."""
        return self.Q.copy()
    
    def get_strategy(self):
        """
        Get current strategy: Epsilon-Greedy mixed strategy.
        Used for metrics calculation.
        """
        strategy = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        max_q = np.max(self.Q)
        best_actions = np.where(self.Q == max_q)[0]
        strategy[best_actions] += (1 - self.epsilon) / len(best_actions)
        return strategy
    
    def reset(self):
        """Reset agent state."""
        super().reset()
        self.Q = np.zeros(self.n_actions)
        self.learning_rate = self.initial_lr
        self.epsilon = self.initial_epsilon

