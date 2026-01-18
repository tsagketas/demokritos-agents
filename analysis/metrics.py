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
    Calculate exploitability: maximum gain from best response.
    
    Args:
        strategy: Current mixed strategy
        game: Game object
        player_id: 0 for Row Player, 1 for Column Player (default: 0)
        
    Returns:
        Exploitability value (float)
    """
    strategy = np.array(strategy)
    
    # Best response to current strategy (for the specified player)
    best_response = game.best_response(strategy, player_id=player_id)
    
    # Expected payoff of best response
    if player_id == 0:
        # Row player: best_response is row action, strategy is column distribution
        expected_payoff_br = np.dot(game.payoff_matrix[best_response], strategy)
        # Expected payoff of current strategy (both players use strategy)
        expected_payoff_current = np.dot(strategy, game.payoff_matrix @ strategy)
    else:
        # Column player: best_response is column action, strategy is row distribution
        # Column player's payoff is negative of row player's payoff
        expected_payoff_br = -np.dot(game.payoff_matrix[:, best_response], strategy)
        expected_payoff_current = -np.dot(strategy, game.payoff_matrix @ strategy)
    
    # Exploitability = difference
    return expected_payoff_br - expected_payoff_current


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

