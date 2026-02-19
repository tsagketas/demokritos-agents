import numpy as np
from .base_agent import BaseAgent


class StochasticQLearningAgent(BaseAgent):
    """
    Q-Learning agent for Stochastic (stateful) games.
    Compatible with the BaseAgent interface while exposing
    additional arguments for state-based updates when needed.
    """

    def __init__(
        self,
        n_actions,
        n_states=81,
        learning_rate=0.1,
        epsilon=0.1,
        lr_decay=0.99995,
        epsilon_decay=0.99995,
        discount_factor=0.95,
        name=None,
    ):
        super().__init__(n_actions, name)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.epsilon_decay = epsilon_decay
        self.gamma = discount_factor

        # Q-table: map state -> array of action values
        self.n_states = n_states
        self.Q = {s: np.zeros(n_actions) for s in range(n_states)}
        self.last_state = 0
        self.player_id = 0  # Default

    def set_player_id(self, player_id):
        self.player_id = player_id

    def act(self, game=None, state=None):
        # Determine current state
        if state is None:
            if hasattr(game, "get_state"):
                state = game.get_state()
            else:
                state = 0

        self.last_state = state

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Exploit current Q-values
        q_values = self.Q[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    def update(self, action, reward, opponent_action=None, next_state=None, done=False):
        """
        Update Q-table.

        Parameters are compatible with BaseAgent.update, while next_state/done
        are optional extras used in stochastic games (e.g. GridGame).
        """
        state = self.last_state

        # Q-learning update:
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        if next_state is None:
            target = reward
        else:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.Q[next_state])

        self.Q[state][action] += self.learning_rate * (target - self.Q[state][action])

        # Decay exploration and learning rate
        self.epsilon *= self.epsilon_decay
        self.learning_rate *= self.lr_decay

        self.reward_history.append(reward)

    def get_strategy(self):
        """
        Approximate current policy at the last visited state as an
        epsilon-greedy mixed strategy over actions.
        """
        q_values = self.Q[self.last_state]
        # If all Q-values are equal (e.g. at init), fall back to uniform
        if np.allclose(q_values, q_values[0]):
            return np.ones(self.n_actions) / self.n_actions

        strategy = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        strategy[best_actions] += (1.0 - self.epsilon) / len(best_actions)
        return strategy

    def reset(self):
        super().reset()
        self.Q = {s: np.zeros(self.n_actions) for s in range(self.n_states)}
