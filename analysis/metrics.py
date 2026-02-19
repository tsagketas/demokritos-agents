import numpy as np


def distance_to_nash(strategy, nash_equilibrium):
    """
    Calculate L2 distance from current strategy to Nash equilibrium.
    
    Args:
        strategy: Current mixed strategy (probability distribution)
        nash_equilibrium: Nash equilibrium strategy
        
    Returns:
        L2 distance (float)
    """
    strategy = np.array(strategy)
    nash_equilibrium = np.array(nash_equilibrium)
    return np.linalg.norm(strategy - nash_equilibrium)


def exploitability(strategy, game, player_id=0):
    """
    Calculate exploitability: how much the OPPONENT can gain by playing best response.
    Exploitability of my strategy = opponent's gain when they exploit me.
    
    Args:
        strategy: Current mixed strategy (of the player we're evaluating)
        game: Game object
        player_id: 0 for Row Player, 1 for Column Player (default: 0)
        
    Returns:
        Exploitability value (float), >= 0. Zero at Nash.
    """
    strategy = np.array(strategy)
    
    # Opponent's best response to current strategy (who exploits us)
    opp_player_id = 1 - player_id
    opp_best_response = game.best_response(strategy, player_id=opp_player_id)
    
    if player_id == 0:
        # Row's strategy: opponent is Column. Column's BR minimizes Row's payoff.
        # Payoff when Column exploits: Row plays strategy, Column plays pure BR
        payoff_when_exploited = np.dot(strategy, game.payoff_matrix[:, opp_best_response])
        payoff_current = np.dot(strategy, game.payoff_matrix @ strategy)
        # Exploitability = how much Row loses when exploited
        return payoff_current - payoff_when_exploited
    else:
        # Column's strategy: opponent is Row. Row's BR maximizes Row's payoff (= minimizes Column's).
        payoff_when_exploited = np.dot(game.payoff_matrix[opp_best_response], strategy)
        payoff_current = np.dot(strategy, game.payoff_matrix @ strategy)
        # Column's payoff = -Row's. Exploitability = Column's loss = Row's gain when exploited
        return payoff_when_exploited - payoff_current


def cumulative_reward(reward_history):
    """
    Calculate total cumulative reward.
    
    Args:
        reward_history: List of rewards received
        
    Returns:
        Cumulative reward (float)
    """
    return sum(reward_history) if reward_history else 0.0


def regret(action_history, reward_history, best_possible_reward):
    """
    Calculate cumulative regret against optimal constant reward (Regret against Nature).
    """
    if not reward_history:
        return 0.0
    
    total_reward = sum(reward_history)
    optimal_reward = best_possible_reward * len(reward_history)
    return optimal_reward - total_reward


def external_regret(agent_history, opponent_history, game, player_id):
    """
    Calculate External Regret: Max gain if we had played the best fixed action 
    against the opponent's actual history, minus actual reward.
    
    Args:
        agent_history: List of actions taken by agent
        opponent_history: List of actions taken by opponent
        game: Game object
        player_id: 0 (Row) or 1 (Col)
        
    Returns:
        External Regret (float)
    """
    if not agent_history or not opponent_history:
        return 0.0
        
    # Calculate actual total reward
    actual_reward = 0
    opponent_history_indices = np.array(opponent_history)
    
    # Calculate expected reward for each fixed action against opponent's history
    # We sum payoffs for playing action 'a' against all 'b' in opponent_history
    
    n_actions = game.n_actions
    cumulative_payoffs_for_fixed_actions = np.zeros(n_actions)
    
    for i, opp_action in enumerate(opponent_history):
        # Calculate actual reward received
        my_action = agent_history[i]
        
        if player_id == 0:
            payoff = game.payoff_matrix[my_action, opp_action]
            # Accumulate potential payoffs for all fixed actions
            cumulative_payoffs_for_fixed_actions += game.payoff_matrix[:, opp_action]
        else:
            # P2 payoff is negative of P1 payoff
            payoff = -game.payoff_matrix[opp_action, my_action]
            # P2 potential payoffs: row i is P2 action, col j is P1 action (opp)
            # P1 matrix is M[opp_action, my_candidate]
            # P2 payoff is -M[opp_action, my_candidate]
            # We want vector of size n_actions (my_candidate)
            cumulative_payoffs_for_fixed_actions += -game.payoff_matrix[opp_action, :]
            
        actual_reward += payoff
        
    # Best fixed strategy payoff
    best_fixed_reward = np.max(cumulative_payoffs_for_fixed_actions)
    
    return best_fixed_reward - actual_reward


def external_regret_history(agent_history, opponent_history, game, player_id):
    """
    Cumulative external regret at each time step (for plotting).
    regret_t = best fixed action payoff up to t - actual reward up to t.
    
    Args:
        agent_history: List of actions taken by agent
        opponent_history: List of actions taken by opponent
        game: Game object
        player_id: 0 (Row) or 1 (Col)
        
    Returns:
        List of cumulative external regrets, length = len(agent_history)
    """
    n = len(agent_history)
    if n == 0 or len(opponent_history) != n:
        return []
    n_actions = game.n_actions
    cumulative_payoffs_per_action = np.zeros(n_actions)
    actual_cumulative = 0
    history = []
    for t in range(n):
        my_action = agent_history[t]
        opp_action = opponent_history[t]
        if player_id == 0:
            actual_cumulative += game.payoff_matrix[my_action, opp_action]
            cumulative_payoffs_per_action += game.payoff_matrix[:, opp_action]
        else:
            actual_cumulative += -game.payoff_matrix[opp_action, my_action]
            cumulative_payoffs_per_action += -game.payoff_matrix[opp_action, :]
        best_fixed = np.max(cumulative_payoffs_per_action)
        history.append(best_fixed - actual_cumulative)
    return history


def strategy_variance(strategy_history):
    """
    Calculate variance in mixed strategy over time.
    
    Args:
        strategy_history: List of strategies (each is a probability distribution)
        
    Returns:
        Variance per action (array) and total variance (float)
    """
    if not strategy_history:
        return np.array([]), 0.0
    
    strategies = np.array(strategy_history)
    variance_per_action = np.var(strategies, axis=0)
    total_variance = np.sum(variance_per_action)
    
    return variance_per_action, total_variance


def convergence_speed(strategy_history, nash_equilibrium, threshold=0.05):
    """
    Calculate iterations to reach threshold distance from Nash.
    
    Args:
        strategy_history: List of strategies over time
        nash_equilibrium: Nash equilibrium strategy
        threshold: Distance threshold (default 0.05 for 95% convergence)
        
    Returns:
        Iteration number when converged (or None if never converged)
    """
    nash_equilibrium = np.array(nash_equilibrium)
    
    for i, strategy in enumerate(strategy_history):
        dist = distance_to_nash(strategy, nash_equilibrium)
        if dist < threshold:
            return i
    
    return None  # Never converged


def strategy_stability(strategy_history, window_size=100):
    """
    Calculate strategy stability: variance over sliding window.
    
    Args:
        strategy_history: List of strategies over time
        window_size: Size of sliding window
        
    Returns:
        List of variance values per iteration
    """
    if not strategy_history:
        return []
    
    variances = []
    for i in range(len(strategy_history)):
        start_idx = max(0, i - window_size + 1)
        window_strategies = strategy_history[start_idx:i+1]
        
        if len(window_strategies) > 1:
            var_per_action, total_var = strategy_variance(window_strategies)
            variances.append(total_var)
        else:
            variances.append(0.0)
    
    return variances


def average_payoff(reward_history):
    """
    Calculate average payoff per round.
    
    Args:
        reward_history: List of rewards received
        
    Returns:
        Average reward (float)
    """
    if not reward_history:
        return 0.0
    return np.mean(reward_history)

