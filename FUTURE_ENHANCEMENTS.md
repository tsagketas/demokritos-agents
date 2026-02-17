# 🚀 Μελλοντικές Προσθήκες & Ενισχύσεις

**Ημερομηνία:** 17 Φεβρουαρίου 2026  
**Στόχος:** Προσθήκη Min-Max Algorithm και μετατροπή Grid Game σε Turn-Based για optimal vs learning σύγκριση

---

## 📊 Τι Έχουμε Ήδη

### Αλγόριθμοι (Agents)
- ✅ **Fictitious Play** (`agents/fictitious_play.py`)
- ✅ **Q-Learning** (`agents/q_learning.py`) - για matrix games
- ✅ **Stochastic Q-Learning** (`agents/stochastic_q_learning.py`) - για grid games

### Παιχνίδια (Games)
- ✅ **Matching Pennies** (`games/matching_pennies.py`) - 2x2 zero-sum
- ✅ **Rock-Paper-Scissors** (`games/rps.py`) - 3x3 zero-sum
- ✅ **Grid Game (Hunter-Prey)** (`games/grid_game.py`) - Stochastic game, simultaneous moves

### Πειράματα (Experiments)
- ✅ **FP vs FP** (`experiments/fp_vs_fp.py`) - Matching Pennies & RPS
- ✅ **FP vs RL** (`experiments/fp_vs_rl.py`) - Matching Pennies & RPS
- ✅ **RL vs RL** (`experiments/rl_vs_rl.py`) - Matching Pennies & RPS ✅ (υπάρχει ήδη)
- ✅ **Grid Game RL vs RL** (`experiments/grid_runner.py`) - Stochastic Q-Learning

---

## 🎯 Τι Θα Προσθέσουμε

### 1. Νέος Αλγόριθμος: Min-Max

**Αρχείο:** `agents/minimax.py`

**Ιδιότητες:**
- Optimal play για deterministic games με perfect information
- Turn-based games (όχι simultaneous)
- Look-ahead με depth limit
- Evaluation function για terminal/non-terminal states

**Χρήση:**
- Baseline για σύγκριση με learning agents
- Optimal outcome σε perfect information games
- Fair comparison όταν και οι δύο παίκτες είναι optimal

**Παράμετροι:**
- `depth`: Πόσα βήματα μπροστά να κοιτάξει (π.χ. depth=5)
- `evaluation_function`: Πώς να αξιολογήσει μια κατάσταση

**Interface:**
```python
class MinimaxAgent(BaseAgent):
    def __init__(self, n_actions, depth=5, evaluation_function=None, name=None):
        # ...
    
    def act(self, game):
        # Min-Max decision
        return best_action
    
    def update(self, action, reward, opponent_action=None):
        # Min-Max doesn't learn, just tracks
        pass
```

---

### 2. Turn-Based Grid Game (Modification)

**Αρχείο:** `games/grid_game.py` (modification ή νέο `turn_based_grid_game.py`)

**Αλλαγές:**
- Αντί για simultaneous moves, turn-based
- Hunter παίζει πρώτος → Prey παίζει δεύτερος → Hunter...
- Perfect information (Hunter βλέπει την κίνηση του Prey)

**Grid Size:**
- **Μόνο 3x3 grid** (size=3)
- `n_states = 81` (3^4 = 81 states)
- Απλότητα και ταχύτητα για demonstrations

**Reward System:**
- **Distance-based reward shaping:** Rewards βασίζονται στη μεταβολή της Manhattan distance
- Hunter: reward = `delta_distance - 0.1` (θετικό αν πλησιάζει)
- Prey: reward = `-delta_distance + 0.1` (θετικό αν απομακρύνεται)
- Terminal rewards: Capture (+10/-10), Timeout (-10/+10)

**Χρήση:**
- Min-Max μπορεί να εφαρμοστεί
- Fair comparison με learning agents

**Interface:**
```python
class TurnBasedGridGame(GridGame):
    def step(self, action):
        # Only one player moves at a time
        # Returns: (reward, done, next_player)
        pass
```

---

### 3. Νέα Πειράματα

#### Πείραμα 6: Min-Max vs Q-Learning (Grid Game - Turn-Based)
**Αρχείο:** `experiments/minmax_vs_rl_grid.py`

**Στόχος:**
- Optimal (Min-Max) vs Learning (Q-Learning) στο Turn-Based Grid Game
- Baseline comparison για optimal play

**Setup:**
- Turn-based Grid Game
- Min-Max Hunter (optimal, depth=3-5)
- Q-Learning Prey (learning)

**Αναμενόμενο Αποτέλεσμα:**
- Min-Max να κερδίζει περισσότερο (optimal play)
- Q-Learning να μαθαίνει να αποφεύγει

**Metrics:**
- Capture rate
- Average steps per capture
- Cumulative rewards
- Learning curve για Q-Learning

---

#### Πείραμα 7: Q-Learning vs Q-Learning (Grid Game - Enhanced)
**Αρχείο:** `experiments/grid_runner.py` (enhancement)

**Βελτιώσεις:**
- Capture rate tracking
- Average steps per capture
- Learning curves για Hunter και Prey
- Strategy visualization

**Metrics:**
- Capture rate over time
- Average steps per capture
- Win rate (Hunter captures vs Prey escapes)

---

## 📋 Προτεραιότητες Υλοποίησης

### Priority 1: Core Components
1. ✅ **Min-Max Agent** (`agents/minimax.py`)
   - Βασική υλοποίηση με depth limit
   - Evaluation function για Grid Game
   - Interface με BaseAgent

2. ✅ **Turn-Based Grid Game** (`games/grid_game.py` - modification)
   - Μετατροπή από simultaneous σε turn-based
   - Perfect information (Hunter βλέπει Prey)
   - Turn order: Hunter → Prey → Hunter...
   - Compatibility με Min-Max

### Priority 2: Experiments
3. ✅ **Min-Max vs Q-Learning (Grid)** (`experiments/minmax_vs_rl_grid.py`)
   - Optimal vs Learning στο Turn-Based Grid Game
   - Learning curve analysis
   - Capture rate tracking

4. ✅ **Enhanced Grid Runner** (`experiments/grid_runner.py`)
   - Περισσότερα metrics για turn-based version
   - Capture rate tracking
   - Learning curves για Hunter και Prey
   - Strategy visualization

---

## 🔍 Τεχνικές Λεπτομέρειες

### Min-Max Algorithm

**Pseudocode:**
```
function minimax(state, depth, isMaximizing):
    if depth == 0 or is_terminal(state):
        return evaluate(state)
    
    if isMaximizing:
        bestValue = -∞
        for each action:
            newState = make_move(state, action)
            value = minimax(newState, depth-1, False)
            bestValue = max(bestValue, value)
        return bestValue
    else:
        bestValue = +∞
        for each action:
            newState = make_move(state, action)
            value = minimax(newState, depth-1, True)
            bestValue = min(bestValue, value)
        return bestValue
```


**Evaluation Function για Grid Game:**
- Terminal states: Capture (+10 για Hunter, -10 για Prey), Escape (-10 για Hunter, +10 για Prey)
- Non-terminal: Heuristic based on distance (π.χ. Manhattan distance από Hunter σε Prey)
- Evaluation: Closer to prey = better for Hunter, farther = better for Prey

**Grid Game Specifications:**
- **Grid Size: 3x3 only** (size=3)
- **State Space: 81 states** (3^4: hunter_row, hunter_col, prey_row, prey_col)
- **Actions: 5** (Up, Right, Down, Left, Stay)
- **State Encoding:** `h_r * 27 + h_c * 9 + p_r * 3 + p_c` (range: 0-80)

**Reward System (Distance-Based):**
- **Capture:** Hunter catches Prey → `+10` για Hunter, `-10` για Prey (game ends)
- **Timeout:** Max steps reached → `-10` για Hunter, `+10` για Prey (game ends)
- **Non-terminal rewards (Reward Shaping):**
  - Υπολογίζει Manhattan distance: `|hunter_row - prey_row| + |hunter_col - prey_col|`
  - `delta_dist = old_distance - new_distance`
  - **Hunter reward:** `delta_dist - 0.1` (time penalty)
    - Αν πλησιάζει (distance ↓) → θετικό reward
    - Αν απομακρύνεται (distance ↑) → αρνητικό reward
  - **Prey reward:** `-delta_dist + 0.1` (zero-sum logic)
    - Αν απομακρύνεται (distance ↑) → θετικό reward
    - Αν πλησιάζει (distance ↓) → αρνητικό reward

---

## 📈 Αναμενόμενα Αποτελέσματα

### Grid Game (Turn-Based)
- **Min-Max vs Q-Learning:** 
  - Min-Max Hunter να έχει υψηλότερο capture rate (optimal play)
  - Q-Learning Prey να μαθαίνει να αποφεύγει μετά από αρκετά episodes
  - Learning curve να δείχνει βελτίωση για Prey
  
- **Q-Learning vs Q-Learning (Turn-Based):** 
  - Learning curves για και τους δύο players
  - Capture rate improvement over time
  - Strategy evolution

---

## 🎓 Ακαδημαϊκή Αξία

### Νέες Συγκρίσεις
1. **Optimal vs Learning:** Min-Max vs Q-Learning στο Turn-Based Grid Game
2. **Learning vs Learning:** Q-Learning vs Q-Learning σε Turn-Based Grid Game

### Νέες Μετρικές
- Capture rate over time
- Average steps per capture
- Learning convergence rates
- Distance from optimal play (Min-Max baseline)

### Νέες Εφαρμογές
- Turn-based games (modification του Grid Game)
- Perfect information για Min-Max
- Optimal play baseline (Min-Max) για Grid Game

---

## ✅ Checklist Υλοποίησης

### Agents
- [ ] `agents/minimax.py` - Min-Max Agent
- [ ] `agents/minimax.py` - Evaluation function για Grid Game

### Games
- [ ] `games/grid_game.py` - Turn-Based modification (ή νέο turn_based_grid_game.py)
- [ ] Turn-based logic (Hunter → Prey → Hunter...)
- [ ] Perfect information support
- [ ] **Grid Size: 3x3 only** (n_states=81, size=3)

### Experiments
- [ ] `experiments/minmax_vs_rl_grid.py` - Min-Max vs Q-Learning (Grid Game)
- [ ] **`experiments/grid_runner.py` - Διορθώσεις**
  - ⚠️ **Grid 3x3:** `size=5` → `size=3`, `n_states=625` → `n_states=81`
  - ⚠️ **Capture heatmap:** Η λίστα `captures` αρχικοποιείται αλλά **δεν γεμίζει ποτέ** (λείπει `captures.append(...)` όταν γίνεται capture)· η `plot_capture_heatmap()` ορίζεται αλλά **δεν καλείται**. Either: (α) στο loop όταν `done` και capture: `captures.append(game.hunter_pos)` (ή prey_pos), μετά κλήση `plot_capture_heatmap(captures, size=3, filename=...)`, ή (β) αφαίρεση dead code (captures, plot_capture_heatmap) αν δεν θέλουμε heatmap
  - Enhanced metrics για turn-based version (όταν γίνει turn-based)

### Analysis
- [ ] Capture rate tracking
- [ ] Average steps per capture
- [ ] Learning curve improvements
- [ ] Strategy visualization για Grid Game
- [ ] **`analysis/gif_maker.py` - Διόρθωση για 3x3 grid**
  - ⚠️ **Hardcoded values:** `GRID_SIZE = 5` → πρέπει `GRID_SIZE = 3`
  - ⚠️ **Hardcoded values:** `N_STATES = 625` → πρέπει `N_STATES = 81`
  - Update όλα τα hardcoded references για 3x3 grid
- [ ] **`analysis/visualizer.py` - Path Organization**
  - Προσθήκη helper function `get_plot_path()` για organized paths
  - Auto-create directories: `os.makedirs(path, exist_ok=True)`
  - Update `ensure_results_dir()` να δημιουργεί nested structure
- [ ] **`experiments/rl_vs_rl.py` - Complete Plots**
  - ⚠️ **Missing plots για RL vs RL:**
    - Matching Pennies: `strategy2.png` (RL2), `distance2.png` (RL2)
    - RPS: `strategy2.png` (RL2), `distance2.png` (RL2), `distance_comparison.png`, `reward_comparison.png`
  - Να είναι consistent με FP vs FP plots (όλα τα plots για και τους δύο agents)

### Results Organization
- [ ] **Οργάνωση Results σε Folders:** `game/(agent vs agent)/photo`
  - Παράδειγμα: `matching_pennies/(FP vs FP)/strategy1.png`
  - Παράδειγμα: `matching_pennies/(RL vs RL)/distance_comparison.png`
  - Παράδειγμα: `rock_paper_scissors/(FP vs RL)/reward_comparison.png`
  - Παράδειγμα: `grid_game/(RL vs RL)/cumulative_reward.png`
  - Παράδειγμα: `grid_game/(MinMax vs RL)/capture_rate.png`
  
- [ ] **Αλλαγές Paths στα Experiments:**
  - ⚠️ **`experiments/fp_vs_fp.py`**: 
    - `results/plots/fp_vs_fp_mp_strategy1.png` → `results/plots/matching_pennies/(FP vs FP)/strategy1.png`
    - `results/plots/fp_vs_fp_rps_strategy1.png` → `results/plots/rock_paper_scissors/(FP vs FP)/strategy1.png`
    - Όλα τα paths για Matching Pennies & RPS
    - **Προαιρετικό (consistency):** RPS λείπει `distance2.png`· προσθήκη `plot_distance_to_nash(results['agent2_distance_history'], 'FP2', ...)` για RPS
  
  - ⚠️ **`experiments/fp_vs_rl.py`**: 
    - `results/plots/fp_vs_rl_mp_*.png` → `results/plots/matching_pennies/(FP vs RL)/*.png`
    - `results/plots/fp_vs_rl_rps_*.png` → `results/plots/rock_paper_scissors/(FP vs RL)/*.png`
    - **Προαιρετικό (consistency):** RPS δεν έχει `avg_payoff` plot· προσθήκη `plot_average_payoff_comparison()` και για RPS
  
  - ⚠️ **`experiments/rl_vs_rl.py`**: 
    - `results/plots/rl_vs_rl_mp_*.png` → `results/plots/matching_pennies/(RL vs RL)/*.png`
    - `results/plots/rl_vs_rl_rps_*.png` → `results/plots/rock_paper_scissors/(RL vs RL)/*.png`
    - `results/plots/rl_exploitability_heatmap.png` → `results/plots/matching_pennies/(RL vs RL)/exploitability_heatmap.png`
    - **Προσθήκη missing plots:**
      - Matching Pennies: `strategy2.png` (RL2), `distance2.png` (RL2)
      - RPS: `strategy2.png` (RL2), `distance2.png` (RL2), `distance_comparison.png`, `reward_comparison.png`
      - Να είναι consistent με FP vs FP plots
  
  - ⚠️ **`experiments/grid_runner.py`**: 
    - `results/plots/grid_cumulative_reward.png` → `results/plots/grid_game/(RL vs RL)/cumulative_reward.png`
    - `results/plots/grid_avg_reward.png` → `results/plots/grid_game/(RL vs RL)/avg_reward.png`
    - Αν προστεθεί κλήση `plot_capture_heatmap()`: path → `results/plots/grid_game/(RL vs RL)/capture_heatmap.png`
  
  - ⚠️ **`analysis/gif_maker.py`**: 
    - `results/plots/hunter_prey_early.gif` → `results/plots/grid_game/(RL vs RL)/hunter_prey_early.gif`
    - `results/plots/hunter_prey_best.gif` → `results/plots/grid_game/(RL vs RL)/hunter_prey_best.gif`

- [ ] **Helper Function (Optional):**
  - Προσθήκη helper function στο `analysis/visualizer.py` για path generation
  - `get_plot_path(game_name, agent1, agent2, plot_name)` → `results/plots/{game}/({agent1} vs {agent2})/{plot_name}.png`
  - Auto-create directories αν δεν υπάρχουν

---

## 📊 Όλες οι Συγκρίσεις με Plots

Πλήρης λίστα experiments που κάνουν σύγκριση και έχουν plots:

### Matrix Games (Matching Pennies & Rock-Paper-Scissors)

| Συγκρίση | Game | Path | Plots |
|----------|------|------|-------|
| **FP vs FP** | Matching Pennies | `matching_pennies/(FP vs FP)/` | strategy1, strategy2, distance1, distance2, distance_comparison, reward_comparison |
| **FP vs FP** | Rock-Paper-Scissors | `rock_paper_scissors/(FP vs FP)/` | strategy1, strategy2, distance1, distance_comparison |
| **FP vs RL** | Matching Pennies | `matching_pennies/(FP vs RL)/` | strategy_fp, strategy_rl, distance_comparison, reward_comparison, avg_payoff |
| **FP vs RL** | Rock-Paper-Scissors | `rock_paper_scissors/(FP vs RL)/` | strategy_fp, strategy_rl, distance_comparison, reward_comparison |
| **RL vs RL** | Matching Pennies | `matching_pennies/(RL vs RL)/` | strategy1, strategy2, distance1, distance2, distance_comparison, reward_comparison, exploitability_heatmap |
| **RL vs RL** | Rock-Paper-Scissors | `rock_paper_scissors/(RL vs RL)/` | strategy1, strategy2, distance1, distance2, distance_comparison, reward_comparison |

### Grid Game (Hunter vs Prey, 3x3)

| Συγκρίση | Path | Plots |
|----------|------|-------|
| **RL vs RL** (υπάρχει) | `grid_game/(RL vs RL)/` | cumulative_reward, avg_reward, hunter_prey_early.gif, hunter_prey_best.gif |
| **MinMax vs RL** (μέλλον) | `grid_game/(MinMax vs RL)/` | capture_rate, learning_curve, cumulative_rewards, avg_steps_per_capture |

### Σύνοψη

- **Matching Pennies:** 3 συγκρίσεις (FP vs FP, FP vs RL, RL vs RL)
- **Rock-Paper-Scissors:** 3 συγκρίσεις (FP vs FP, FP vs RL, RL vs RL)
- **Grid Game:** 2 συγκρίσεις (RL vs RL, MinMax vs RL)

**Συνολικά:** 8 experiment configurations με plots.

---

## 📁 Οργάνωση Results

### Δομή Φακέλων για Results
Όλα τα results (plots, visualizations) πρέπει να οργανώνονται σε φακέλους με τη δομή:
```
results/
  plots/
    matching_pennies/
      (FP vs FP)/
        strategy1.png
        strategy2.png
        distance_comparison.png
        reward_comparison.png
      (FP vs RL)/
        strategy_fp.png
        strategy_rl.png
        distance_comparison.png
      (RL vs RL)/
        strategy1.png
        distance_comparison.png
        reward_comparison.png
    rock_paper_scissors/
      (FP vs FP)/
        ...
      (FP vs RL)/
        ...
      (RL vs RL)/
        ...
    grid_game/
      (RL vs RL)/
        cumulative_reward.png
        avg_reward.png
        capture_heatmap.png
      (MinMax vs RL)/
        capture_rate.png
        learning_curve.png
        ...
```

**Format:** `game_name/(agent1 vs agent2)/plot_name.png`

**Πρέπει να ενημερωθούν:**
- `experiments/fp_vs_fp.py` - Matching Pennies & RPS
- `experiments/fp_vs_rl.py` - Matching Pennies & RPS
- `experiments/rl_vs_rl.py` - Matching Pennies & RPS
- `experiments/grid_runner.py` - Grid Game
- Όλα τα νέα experiments

---

## 📝 Σημειώσεις

### Γιατί Turn-Based Grid Game:
- **Min-Max Compatibility:** Min-Max χρειάζεται turn-based games
- **Fair Comparison:** Perfect information για Min-Max
- **Baseline:** Optimal vs Learning comparison
- **Realistic Scenario:** Turn-based games είναι πιο κοινά από simultaneous
- **Learning Challenge:** Q-Learning να μάθει να ανταγωνίζεται optimal play

### Γιατί Min-Max στο Grid Game:
- **Optimal Baseline:** Δείχνει το best possible outcome
- **Evaluation Function:** Distance-based heuristic για non-terminal states
- **Depth Limit:** Grid Game έχει μεγάλο state space, depth limit είναι απαραίτητο
- **Comparison:** Fair comparison με learning agents

---

**Τελευταία Ενημέρωση:** 17 Φεβρουαρίου 2026  
**Status:** Planning Phase - Ready for Implementation  
**Αλλαγές:** 
- Tic-Tac-Toe ακυρώθηκε, εστίαση σε Turn-Based Grid Game με Min-Max
- RL vs RL υπάρχει ήδη για Matching Pennies & RPS ✅
- Results organization: `game/(agent vs agent)/photo` structure για όλα τα experiments
