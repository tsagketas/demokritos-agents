# 📋 Προετοιμασία Συνάντησης με τον Καθηγητή Γρηγόρο

**Ημερομηνία:** 28 Ιανουαρίου 2026  
**Θέμα:** Fictitious Play & Reinforcement Learning for Computing Equilibria

---

## 1️⃣ Τι Έχουμε Κάνει (Για Παρουσίαση)

### Αλγόριθμοι Υλοποιημένοι

#### ✅ **Fictitious Play (FP)** - `agents/fictitious_play.py`

**Πώς Λειτουργεί:**
1. **Initialization:** Ξεκινάει με uniform belief (1/n για κάθε ενέργεια)
2. **Belief Update:** Κάθε iteration:
   - Παρακολουθεί την ενέργεια του αντιπάλου
   - Μετράει πόσες φορές έπαιξε κάθε ενέργεια (`action_counts`)
   - Υπολογίζει τη μέση στρατηγική: `belief = action_counts / total_actions`
   - **Incremental update (O(1))** - δεν χρειάζεται να υπολογίζει από την αρχή
3. **Action Selection:** Παίζει **best response** στη current belief (μέσω `game.best_response()`)
4. **Tracking:** Κρατάει ιστορία: `opponent_history`, `action_history`, `reward_history`

**Τεχνικές Λεπτομέρειες:**
- Χρησιμοποιεί incremental updates (O(1) complexity)
- Belief = empirical frequency distribution του αντιπάλου
- Best response υπολογίζεται από το game object (χρησιμοποιεί payoff matrix)

**Αποτελέσματα:**
- Matching Pennies: Σύγκλιση σε **5 iterations** (distance to Nash = 0.0007)
- Rock-Paper-Scissors: Σύγκλιση σε **72 iterations** (distance = 0.0032)
- Δείχνει Shapley polygons στο RPS (ταλαντώσεις πριν τη σύγκλιση)

---

#### ✅ **Q-Learning (RL)** - `agents/q_learning.py`

**Πώς Λειτουργεί:**
1. **Initialization:** 
   - Q-table: `Q = zeros(n_actions)` (ένα Q-value για κάθε ενέργεια)
   - Learning rate: `α = 0.1` (default)
   - Epsilon: `ε = 0.1` (default, για exploration)
   - Decay parameters: `lr_decay = 0.99995`, `epsilon_decay = 0.99995`

2. **Action Selection (Epsilon-Greedy):**
   - Με πιθανότητα `ε`: **Explore** (τυχαία ενέργεια)
   - Με πιθανότητα `1-ε`: **Exploit** (ενέργεια με max Q-value)
   - Αν υπάρχουν ties, επιλέγει τυχαία μεταξύ τους

3. **Q-Value Update:**
   - `Q[action] = Q[action] + α * (reward - Q[action])`
   - Αυτό είναι **bandit-like update** (δεν χρειάζεται next state για zero-sum games)

4. **Parameter Decay:**
   - `epsilon = max(min_epsilon, epsilon * epsilon_decay)` (μειώνεται με τον χρόνο)
   - `learning_rate = max(min_lr, learning_rate * lr_decay)` (μειώνεται με τον χρόνο)

5. **Strategy Extraction:**
   - Για metrics: `strategy = epsilon/n * uniform + (1-epsilon) * one_hot(best_action)`
   - Αυτό δίνει mixed strategy από epsilon-greedy policy

**Τεχνικές Λεπτομέρειες:**
- Χρησιμοποιεί bandit-like Q-learning (no discount factor γ, no next-state max Q)
- Είναι κατάλληλο για repeated games όπου δεν υπάρχει "next state"
- Epsilon decay εξασφαλίζει transition από exploration σε exploitation

**Αποτελέσματα:**
- FP vs RL (Matching Pennies): Κερδίζει (+374 points) αλλά δεν συγκλίνει (distance = 0.6642)
- FP vs RL (RPS): Κερδίζει (+213 points) αλλά δεν συγκλίνει (distance = 0.7670)
- **Key Finding:** Εκμεταλλεύεται την προβλεψιμότητα του FP

---

#### ✅ **Stochastic Q-Learning** - `agents/stochastic_q_learning.py`

**Πώς Λειτουργεί:**
1. **State-Action Q-Table:**
   - `Q[state][action]` για κάθε state και action
   - Για Grid Game: `n_states = 81` (3x3 grid: 3^4 = 81 states)
   - State encoding: `state = h_row * size³ + h_col * size² + p_row * size + p_col`

2. **Action Selection:**
   - Epsilon-greedy όπως το standard Q-Learning
   - Αλλά εξαρτάται από το current state: `Q[state][action]`

3. **Q-Update:**
   - `Q[state][action] = Q[state][action] + α * (reward - Q[state][action])`
   - Simplified update (no γ * max Q(s', a') - κατάλληλο για zero-sum context)

4. **Grid Game Specifics:**
   - 5 actions: Up, Right, Down, Left, Stay
   - State tracking: `game.get_state()` επιστρέφει encoded state
   - Rewards: Hunter (+10 capture, -1 step), Prey (-10 capture, +1 step)

**Τεχνικές Λεπτομέρειες:**
- Q-table: Dictionary `{state: array(n_actions)}`
- State encoding: Flattened 4D position (hunter_row, hunter_col, prey_row, prey_col)
- Simplified update (bandit-like per state) - no discount factor

**Αποτελέσματα:**
- Grid Game (3x3): Θετική κλίση στο cumulative reward
- Μετά από ~10,000 iterations, average reward γίνεται θετικός
- Hunter μαθαίνει να εγκλωβίζει το Prey

---

### Παιχνίδια Υλοποιημένα

#### ✅ **Matching Pennies** - `games/matching_pennies.py`

**Payoff Matrix (Row Player):**
```
        H    T
    H  [+1, -1]
    T  [-1, +1]
```

**Ιδιότητες:**
- Zero-sum game (αν ο row παίρνει +1, ο column παίρνει -1)
- 2 ενέργειες: H (Heads=0), T (Tails=1)
- **Nash Equilibrium:** (0.5, 0.5) - uniform mixed strategy
- **Theoretical:** Καμία pure strategy Nash, μόνο mixed

**Υλοποίηση:**
- Payoff matrix: `[[1, -1], [-1, 1]]`
- `get_nash_equilibrium()`: Επιστρέφει `[0.5, 0.5]`
- `get_action_name()`: Μετατρέπει 0→'H', 1→'T'

---

#### ✅ **Rock-Paper-Scissors (RPS)** - `games/rps.py`

**Payoff Matrix (Row Player):**
```
        R    P    S
    R  [0,  -1,  +1]
    P  [+1,  0,  -1]
    S  [-1, +1,   0]
```

**Ιδιότητες:**
- Zero-sum game
- 3 ενέργειες: R (Rock=0), P (Paper=1), S (Scissors=2)
- Κυκλικό: R beats S, S beats P, P beats R
- **Nash Equilibrium:** (1/3, 1/3, 1/3) - uniform mixed strategy
- **Theoretical:** Shapley polygons (ταλαντώσεις πριν τη σύγκλιση)

**Υλοποίηση:**
- Payoff matrix: `[[0, -1, 1], [1, 0, -1], [-1, 1, 0]]`
- `get_nash_equilibrium()`: Επιστρέφει `[1/3, 1/3, 1/3]`
- `get_action_name()`: Μετατρέπει 0→'R', 1→'P', 2→'S'

---

#### ✅ **Grid Game (Hunter vs Prey)** - `games/grid_game.py`

**Setup:**
- Grid size: 3x3 (default, αλλά μπορεί να αλλάξει)
- State space: 81 states (3^4 = hunter_row, hunter_col, prey_row, prey_col)
- Actions: 5 (Up=0, Right=1, Down=2, Left=3, Stay=4)

**Κανόνες:**
- **Hunter (Player 1):**
  - +10 αν πιάσει Prey (same cell)
  - -1 ανά βήμα (energy cost)
- **Prey (Player 2):**
  - -10 αν πιαστεί
  - +1 ανά βήμα (survival bonus)
- **Zero-sum:** Αν Hunter παίρνει +10, Prey παίρνει -10

**State Encoding:**
- `state = h_row * size³ + h_col * size² + p_row * size + p_col`
- Για 3x3: max state = 80

**Movement:**
- Simultaneous movement (και οι δύο κινούνται ταυτόχρονα)
- Bounds checking (δεν μπορούν να βγουν έξω από το grid)
- Αν collision: Reset positions σε (0,0) και (size-1, size-1)

**Υλοποίηση:**
- `step(action1, action2)`: Εκτελεί moves και επιστρέφει rewards
- `get_state()`: Επιστρέφει encoded state
- `_move(pos, action)`: Υπολογίζει νέα θέση με bounds checking

---

### Πειράματα & Αποτελέσματα

#### ✅ **Πείραμα 1: FP vs FP (Matching Pennies)**

**Setup:**
- 2 Fictitious Play agents
- 10,000 iterations
- Seed: 42 (reproducible)

**Αποτελέσματα:**
- **FP1 (Row Player):**
  - Final distance to Nash: **0.0007** (σχεδόν τέλεια)
  - Convergence iteration: **5** (εξαιρετικά γρήγορη)
  - Cumulative reward: **+60.00**
  - Average reward: **0.0060**
- **FP2 (Column Player):**
  - Final distance to Nash: **0.0092**
  - Convergence iteration: **1** (άμεση σύγκλιση)
  - Cumulative reward: **-60.00** (zero-sum)
  - Average reward: **-0.0060**

**Γραφήματα Παράγονται:**
- `fp_vs_fp_mp_strategy1.png` - Strategy evolution για FP1
- `fp_vs_fp_mp_strategy2.png` - Strategy evolution για FP2
- `fp_vs_fp_mp_distance1.png` - Distance to Nash για FP1
- `fp_vs_fp_mp_distance2.png` - Distance to Nash για FP2
- `fp_vs_fp_mp_distance_comparison.png` - Comparison distance
- `fp_vs_fp_mp_reward_comparison.png` - Comparison cumulative rewards

**Συμπέρασμα:** Ο FP λειτουργεί άψογα σε απλά παιχνίδια - σύγκλιση σε 5 iterations!

---

#### ✅ **Πείραμα 2: FP vs FP (Rock-Paper-Scissors)**

**Setup:**
- 2 Fictitious Play agents
- 10,000 iterations
- Seed: 42

**Αποτελέσματα:**
- **FP1 & FP2:**
  - Final distance to Nash: **0.0032** (πολύ καλή σύγκλιση)
  - Convergence iteration: **72** (αργότερα από Matching Pennies)
  - Cumulative reward: **0.00** (ισοπαλία, όπως αναμενόταν)
  - Average reward: **0.0000**

**Γραφήματα Παράγονται:**
- `fp_vs_fp_rps_strategy1.png` - Strategy evolution (δείχνει Shapley polygons)
- `fp_vs_fp_rps_strategy2.png` - Strategy evolution
- `fp_vs_fp_rps_distance1.png` - Distance to Nash
- `fp_vs_fp_rps_distance_comparison.png` - Comparison

**Συμπέρασμα:** Ο FP συγκλίνει αλλά χρειάζεται περισσότερο χρόνο σε κυκλικά παιχνίδια. Δείχνει Shapley polygons (ταλαντώσεις πριν τη σύγκλιση).

---

#### ✅ **Πείραμα 3: FP vs RL (Matching Pennies)**

**Setup:**
- FP agent vs Q-Learning agent
- RL parameters: `learning_rate=0.1`, `epsilon=0.1`
- 10,000 iterations
- Seed: 42

**Αποτελέσματα:**
- **FP:**
  - Final distance to Nash: **0.0000** (τέλεια σύγκλιση!)
  - Convergence iteration: **5**
  - Cumulative reward: **-374.00** (Ηττημένος)
  - Average reward: **-0.0374**
- **RL:**
  - Final distance to Nash: **0.6642** (δεν συγκλίνει)
  - Convergence: **None** (δεν συγκλίνει)
  - Cumulative reward: **+374.00** (Νικητής!)
  - Average reward: **+0.0374**

**Γραφήματα Παράγονται:**
- `fp_vs_rl_mp_strategy_fp.png` - FP strategy evolution
- `fp_vs_rl_mp_strategy_rl.png` - RL strategy evolution
- `fp_vs_rl_mp_distance_comparison.png` - Distance comparison
- `fp_vs_rl_mp_reward_comparison.png` - Reward comparison
- `fp_vs_rl_mp_avg_payoff.png` - Average payoff comparison

**Συμπέρασμα:** Το πιο ενδιαφέρον finding! Ο FP συγκλίνει τέλεια στο Nash αλλά **χάνει**. Ο RL δεν συγκλίνει αλλά **κερδίζει** εκμεταλλευόμενος την προβλεψιμότητα του FP.

---

#### ✅ **Πείραμα 4: FP vs RL (Rock-Paper-Scissors)**

**Setup:**
- FP agent vs Q-Learning agent
- RL parameters: `learning_rate=0.1`, `epsilon=0.1`
- 10,000 iterations
- Seed: 42

**Αποτελέσματα:**
- **FP:**
  - Final distance to Nash: **0.0152** (καλή σύγκλιση)
  - Convergence iteration: **37**
  - Cumulative reward: **-213.00** (Ηττημένος)
  - Average reward: **-0.0213**
- **RL:**
  - Final distance to Nash: **0.7670** (δεν συγκλίνει)
  - Convergence: **0** (δεν συγκλίνει)
  - Cumulative reward: **+213.00** (Νικητής!)
  - Average reward: **+0.0213**

**Γραφήματα Παράγονται:**
- `fp_vs_rl_rps_strategy_fp.png` - FP strategy evolution
- `fp_vs_rl_rps_strategy_rl.png` - RL strategy evolution
- `fp_vs_rl_rps_distance_comparison.png` - Distance comparison
- `fp_vs_rl_rps_reward_comparison.png` - Reward comparison

**Συμπέρασμα:** Το pattern επιβεβαιώνεται - σε adaptive environments, ο RL υπερέχει παρά το ότι δεν συγκλίνει στο Nash.

---

#### ✅ **Πείραμα 5: Grid Game (Hunter vs Prey)**

**Setup:**
- 2 Stochastic Q-Learning agents (Hunter vs Prey)
- Grid size: 5x5 (625 states)
- Hunter: `learning_rate=0.1`, `epsilon=0.5` (high exploration), `lr_decay=0.99999`
- Prey: `learning_rate=0.1`, `epsilon=0.5`, `lr_decay=0.99999`
- 200,000 iterations (πολύ περισσότερα για spatial learning)

**Αποτελέσματα:**
- **Hunter Cumulative Reward:** Θετική κλίση (απόδειξη μάθησης)
- **Average Reward:** Μετά από ~10,000 iterations, γίνεται θετικός
- **Learning Curve:** Η γραμμή ανεβαίνει συνεχώς (Hunter μαθαίνει να πιάνει Prey)

**Γραφήματα Παράγονται:**
- `grid_cumulative_reward.png` - Cumulative reward του Hunter
- `grid_avg_reward.png` - Moving average reward (window=1000)

**Συμπέρασμα:** Ο RL μπορεί να λύσει και spatial problems - μαθαίνει strategies σε grid world.

---

### Μετρικές & Ανάλυση

#### ✅ **Μετρικές Υλοποιημένες** - `analysis/metrics.py`

1. **Distance to Nash:**
   - L2 distance: `||strategy - nash_equilibrium||`
   - Μετράει πόσο κοντά είναι η current strategy στο Nash

2. **Exploitability:**
   - `exploitability = payoff(best_response) - payoff(current_strategy)`
   - Μετράει πόσο "εκμεταλλεύσιμη" είναι μια στρατηγική

3. **Cumulative Reward:**
   - `sum(reward_history)`
   - Συνολικό κέρδος/ζημιά

4. **Average Reward:**
   - `mean(reward_history)`
   - Μέσος όρος ανά iteration

5. **Strategy Evolution:**
   - Track πώς αλλάζει η mixed strategy με τον χρόνο
   - Χρήσιμο για να δούμε Shapley polygons

6. **Convergence Speed:**
   - `convergence_speed(strategy_history, nash, threshold=0.05)`
   - Επιστρέφει iteration όταν έφτασε threshold distance

7. **Strategy Stability:**
   - Variance σε sliding window
   - Μετράει πόσο σταθερή είναι η στρατηγική

8. **External Regret:**
   - Υλοποιημένο αλλά δεν χρησιμοποιείται στα main experiments
   - Μετράει regret vs best fixed action

#### ✅ **Οπτικοποίηση** - `analysis/visualizer.py`

**Γραφήματα που παράγονται:**
- Strategy evolution plots (πώς αλλάζουν οι πιθανότητες με τον χρόνο)
- Distance to Nash plots (convergence curves)
- Cumulative reward comparisons
- Average payoff comparisons
- Multiple agents comparison plots

**Αποθηκευμένα:**
- Όλα τα plots σε `results/plots/`
- PNG format, publication-ready
- Clear labels, legends, titles


---

## 2️⃣ Ερωτήσεις για το Πρακτικό Μέρος

### 🔵 **Ερώτηση 1: Εύρος Grid Game**
> "Στο Grid Game (Hunter vs Prey) έχω υλοποιήσει ένα 3x3 grid όπου φαίνεται ξεκάθαρα η μάθηση (θετική κλίση στο reward, learning μετά από ~10k iterations).
>
> **Θεωρείτε ότι το 3x3 είναι επαρκές για να αποδείξω το spatial learning, ή θα προσέφερε ουσιαστική αξία στην εργασία να το μεγαλώσω (π.χ. σε 5x5);**
>
> *(Σημείωση: Ανησυχώ μήπως μεγαλώσει εκθετικά ο χρόνος εκπαίδευσης χωρίς να αλλάξει το ποιοτικό συμπέρασμα).*"

---

### 🔵 **Ερώτηση 2: Στατιστική Εγκυρότητα**
> "Τώρα τρέχω κάθε πείραμα μία φορά με ένα seed. **Θα ήταν σημαντικό να τρέξω πολλαπλές φορές (10-20 seeds) και να δείξω mean ± standard deviation για στατιστική εγκυρότητα, ή τα τρέχοντα αποτελέσματα είναι επαρκή;**"

---

### 🔵 **Ερώτηση 3: Hyperparameter Analysis**
> "Έχω υλοποιήσει Q-Learning με learning rate και epsilon decay. **Θα ήταν χρήσιμο να κάνω hyperparameter sweep (διαφορετικές τιμές learning rate, epsilon) και να δείξω πώς επηρεάζουν τα αποτελέσματα, ή να εστιάσω στην ανάλυση των τρέχοντας αποτελεσμάτων;**"

---

### 🔵 **Ερώτηση 4: Επέκταση σε General-Sum Games**
> "Αυτή τη στιγμή όλα τα παιχνίδια είναι Zero-Sum (ανταγωνιστικά). Επειδή ο κώδικας μου επιτρέπει εύκολη επέκταση,
>
> **Θα είχε νόημα να τρέξω κι ένα σενάριο General-Sum (π.χ. Prisoner's Dilemma) για να δούμε αν συνεργάζονται, ή να εστιάσω στην τελειοποίηση της ανάλυσης των Zero-Sum που έχω ήδη;**"

---

### 🔵 **Ερώτηση 5: Ερμηνεία Αποτελεσμάτων FP vs RL**
> "Στο πείραμα FP vs RL παρατηρώ ότι ο Q-Learning agent κερδίζει συνεχώς τον Fictitious Play, αλλά **δεν** συγκλίνει στο Nash Equilibrium.
>
> Το ερμηνεύω ως εξής: Ο RL κάνει 'exploitation' την στασιμότητα του FP (που υποθέτει στατικό αντίπαλο) αντί να ψάχνει την ισορροπία.
>
> **Συμφωνείτε με αυτή την ερμηνεία, ή πιστεύετε ότι πρέπει να ψάξω και για θέματα παραμέτρων (hyperparameters) ή άλλες εξηγήσεις;**"

---

### 🔵 **Ερώτηση 6: Comparison με Theoretical Nash**
> "Έχω υπολογίσει distance to Nash για κάθε agent. **Θα ήταν σημαντικό να συμπεριλάβω στο report έναν πίνακα με theoretical vs experimental Nash values και να αναλύσω τις διαφορές, ή αρκεί η distance metric;**"

---

## 3️⃣ Ερωτήσεις για το Θεωρητικό Μέρος

### 🔵 **Ερώτηση 1: Θεωρητικό Βάθος στο Report**
> "Στο Report, πόσο βάθος θέλετε να δώσω στη μαθηματική απόδειξη σύγκλισης του Fictitious Play (Brown 1951);
>
> **Να παραθέσω απλά το θεώρημα και να εστιάσω στα πειραματικά αποτελέσματα, ή θέλετε να δώσω έμφαση και στην απόδειξη;**
>
> Επίσης, πόσο λεπτομερώς να περιγράψω τους αλγόριθμους; Να συμπεριλάβω pseudocode ή αρκεί η περιγραφή με λόγια;"

---

### 🔵 **Ερώτηση 2: Μαθηματική Διατύπωση**
> "**Πόσο λεπτομερής να είναι η μαθηματική διατύπωση;** Να συμπεριλάβω:
> - Τυπική διατύπωση zero-sum games (payoff matrices, strategies)
> - Μαθηματική διατύπωση FP (belief updates, best response)
> - Μαθηματική διατύπωση Q-Learning (Bellman equation, Q-updates)
>
> **Ή αρκεί μια πιο high-level περιγραφή;**"

---

### 🔵 **Ερώτηση 3: Algorithm Pseudocode**
> "**Να συμπεριλάβω formal pseudocode για τους αλγόριθμους;** 
> - Pseudocode για Fictitious Play
> - Pseudocode για Q-Learning
> - Pseudocode για Stochastic Q-Learning
>
> **Ή αρκεί η περιγραφή με λόγια και τα βήματα;**"

---

### 🔵 **Ερώτηση 4: Θεωρήματα & References**
> "**Ποια θεωρήματα να συμπεριλάβω;**
> - Brown (1951) - Σύγκλιση FP
> - Shapley (1964) - Shapley Polygons
> - Watkins & Dayan (1992) - Q-Learning convergence
>
> **Να παραθέσω τα θεωρήματα με απόδειξη, ή αρκεί η αναφορά και η ερμηνεία;**"

---

### 🔵 **Ερώτηση 5: Θεωρητική Σύγκριση FP vs RL**
> "**Πόσο λεπτομερής να είναι η θεωρητική σύγκριση FP vs RL;**
> - Computational complexity (time/space)
> - Convergence properties
> - Assumptions (static vs adaptive opponents)
>
> **Ή να εστιάσω στην πειραματική σύγκριση;**"

---

### 🔵 **Ερώτηση 6: Μορφή Report**
> "**Τι μορφή προτιμάτε για το report;** 
> - LaTeX PDF
> - Markdown
> - Word
>
> **Και πόσες σελίδες περίπου;**"

---

## 4️⃣ Γενικές Ερωτήσεις (Για Ολόκληρη την Εργασία)

### 🔵 **Ερώτηση 1: Scope & Focus**
> "Έχω υλοποιήσει:
> - FP και RL για zero-sum games
> - 3 διαφορετικά παιχνίδια (MP, RPS, Grid)
> - 5 πειράματα με συγκρίσεις
>
> **Θεωρείτε ότι αυτό είναι επαρκές για Master's level (2 άτομα, 30% theory, 70% implementation), ή χρειάζεται κάτι επιπλέον;**"

---

### 🔵 **Ερώτηση 2: Κύρια Findings**
> "Το κύριο finding μου είναι ότι σε adaptive environments, ο RL υπερέχει παρά το ότι δεν συγκλίνει στο Nash, ενώ ο FP συγκλίνει αλλά χάνει.
>
> **Είναι αυτό ένα σημαντικό contribution για το report, ή να εστιάσω περισσότερο στην σύγκλιση και τα theoretical properties;**"

---

### 🔵 **Ερώτηση 3: Παρουσίαση**
> "**Για την παρουσίαση:**
> - Πόσες διαφάνειες περίπου;
> - Τι μορφή: PowerPoint, Beamer (LaTeX), ή άλλη;
> - Πόσο χρόνο έχουμε;
> - Τι να εστιάσω: theory, results, ή και τα δύο;"

---

### 🔵 **Ερώτηση 4: Code Documentation**
> "**Πόσο λεπτομερής να είναι η τεκμηρίωση του κώδικα;**
> - Docstrings για κάθε function
> - README με instructions
> - Comments στο code
>
> **Τι επίπεδο περιμένετε;**"

---

### 🔵 **Ερώτηση 5: Reproducibility**
> "Έχω Docker setup και seeds για reproducibility. **Είναι επαρκές, ή χρειάζεται κάτι επιπλέον (π.χ. requirements.txt, detailed README, environment setup instructions);**"

---

### 🔵 **Ερώτηση 6: Missing Elements**
> "**Υπάρχει κάτι που λείπει από την εργασία που θα έπρεπε να έχω;**
> - Regret analysis (έχω κώδικα αλλά δεν τον χρησιμοποιώ)
> - Computational complexity analysis
> - Περισσότερα παιχνίδια
> - Άλλο;"

---

### 🔵 **Ερώτηση 7: Evaluation Criteria**
> "**Πώς θα αξιολογηθεί η εργασία;**
> - Τι είναι τα κριτήρια;
> - Τι βάρος έχει το implementation vs theory;
> - Τι βάρος έχει το report vs presentation;
>
> **Για να ξέρω πού να εστιάσω;**"

---

### 🔵 **Ερώτηση 8: Timeline**
> "**Ποια είναι η deadline για:**
> - Report submission
> - Presentation
> - Code submission
>
> **Και υπάρχει κάποιο intermediate deadline;**"

---

### 🔵 **Ερώτηση 9: Collaboration**
> "Είμαστε 2 άτομα. **Πώς να οργανώσουμε τη δουλειά;**
> - Να χωρίσουμε theory vs implementation;
> - Να κάνουμε και οι δύο και τα δύο;
> - Πώς να συντονιστούμε για το report;
>
> **Έχετε συμβουλές;**"

---

### 🔵 **Ερώτηση 10: Quality Check**
> "**Πριν την τελική υποβολή, θέλετε να σας στείλω draft για feedback, ή να υποβάλω απευθείας;**"

---

