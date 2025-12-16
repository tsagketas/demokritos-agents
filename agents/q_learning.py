import numpy as np
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Q-Learning agent: learns action values through exploration and exploitation."""
    
    def __init__(self, n_actions, learning_rate=0.1, epsilon=0.1, name=None):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_actions: Number of available actions
            learning_rate: Learning rate (alpha) for Q-value updates
            epsilon: Exploration rate (probability of random action)
            name: Optional name for the agent
        """
        super().__init__(n_actions, name)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.Q = np.zeros(n_actions)  # Q-values for each action
    
    def act(self, game=None):
        """
        Choose action: epsilon-greedy policy.
        
        Args:
            game: Game object (not used in Q-learning, but kept for interface consistency)
            
        Returns:
            Action index (exploit with probability 1-epsilon, explore with probability epsilon)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action according to Q-values
            # If multiple actions have same Q-value, choose randomly among them
            max_q = np.max(self.Q)
            best_actions = np.where(self.Q == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, action, reward, opponent_action=None):
        """
        Update Q-value for the action taken.
        
        Args:
            action: Action the agent took
            reward: Reward received
            opponent_action: Opponent's action (not used in Q-learning)
        """
        # Q-learning update: Q(a) = Q(a) + alpha * (reward - Q(a))
        self.Q[action] += self.learning_rate * (reward - self.Q[action])
        
        # Track history
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def get_q_values(self):
        """Get current Q-values."""
        return self.Q.copy()
    
    def get_strategy(self):
        """
        Get current strategy: softmax over Q-values (for visualization).
        Temperature parameter controls exploration.
        """
        # Use softmax with temperature = 1.0
        # Higher Q-values get higher probabilities
        exp_q = np.exp(self.Q - np.max(self.Q))  # Subtract max for numerical stability
        return exp_q / np.sum(exp_q)
    
    def reset(self):
        """Reset agent state."""
        super().reset()
        self.Q = np.zeros(self.n_actions)

