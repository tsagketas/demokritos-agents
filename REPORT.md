# Αναφορά Εργασίας: Fictitious play and reinforcement learning for computing equilibria

Ημερομηνία: Φεβρουάριος 2026  

---

## Περίληψη

Στα πλαίσια της παρούσας εργασίας μελετήσαμε τη συμπεριφορά πρακτόρων σε ανταγωνιστικά περιβάλλοντα. Υλοποιήσαμε και συγκρίναμε δύο βασικές προσεγγίσεις μάθησης: το Fictitious Play (FP), που βασίζεται στη μοντελοποίηση του αντιπάλου μέσω εμπειρικής κατανομής και best response, και το Reinforcement Learning (Q-learning), που μαθαίνει αξίες ενεργειών χωρίς ρητό μοντέλο του αντιπάλου, καθώς και τον αλγόριθμο Minimax για στοχαστικά turn based παίγνια. Πραγματοποιήσαμε επίσης σύγκριση Minimax και RL στο Grid Hunter-Prey (Hunter με Minimax, Prey με Q-learning). Τα πειράματα διεξήχθησαν σε τρία παίγνια: Matching Pennies, Rock-Paper-Scissors και Grid Hunter-Prey (στοχαστικό, turn-based).

---

## 1. Θεωρητικό Πλαίσιο και Στόχοι

Η εργασία ξεκινά από το θεωρητικό πλαίσιο της θεωρίας παιγνίων (normal-form, zero-sum, Nash equilibrium, repeated games) και της μάθησης πρακτόρων (Fictitious Play, Q-learning). Στόχος είναι η κατανόηση σε βάθος και η εφαρμογή σε απλά, καλά ορισμένα προβλήματα, χωρίς να προστεθεί περιττή πολυπλοκότητα. Τα παραδείγματα επιλέχθηκαν με πληροφορημένο τρόπο ώστε να φαίνονται ξεκάθαρα τόσο οι δυνατότητες όσο και οι αδυναμίες των αλγορίθμων: π.χ. ότι η FP συγκλίνει στο Nash αλλά μπορεί να γίνει ευάλωτη σε προσαρμοστικούς αντιπάλους, ότι το Q-learning μπορεί να εκμεταλλευτεί προβλεψιμότητα αλλά στο self-play δεν συγκλίνει απαραίτητα σε ισορροπία, και ότι η Minimax υπερτερεί όταν το μοντέλο του κόσμου είναι γνωστό. Αυτή η επιλογή απλών αλλά εκφραστικών παραδειγμάτων είναι στο κέντρο της μεθοδολογίας της εργασίας.

### 1.1 Πράκτορες και Χρησιμότητα

Οι πράκτορες ορίζονται ως ορθολογικές οντότητες που στοχεύουν στη μεγιστοποίηση της Αναμενόμενης Χρησιμότητας (Maximum Expected Utility, MEU). Σε περιβάλλοντα πολλαπλών πρακτόρων η χρησιμότητα ενός πράκτορα εξαρτάται και από τις ενέργειες των άλλων, οδηγώντας σε στρατηγική σκέψη και σε έννοιες όπως η ισορροπία Nash.

### 1.2 Θεωρία Παιγνίων

Εστιάζουμε σε παίγνια μηδενικού αθροίσματος (zero-sum), όπου U₁ + U₂ = 0. Σε τέτοια παίγνια ο καθαρός ανταγωνισμός μοντελοποιείται με πίνακες αποδόσεων (payoff matrices). Η Ισορροπία Nash (NE) είναι το σημείο όπου κανένας παίκτης δεν έχει κίνητρο να αλλάξει μονομερώς τη στρατηγική του. Στα παίγνια που επιλέξαμε (Matching Pennies, Rock-Paper-Scissors) δεν υπάρχει Pure Strategy NE, αλλά υπάρχει μοναδικό Mixed Strategy NE. Σε zero-sum παίγνια το θεώρημα Minimax εξασφαλίζει ότι η λύση minimax ταυτίζεται με το Nash Equilibrium.

### 1.3 Μάθηση σε Επαναλαμβανόμενα Παίγνια

Σε repeated games εξετάζουμε δύο προσεγγίσεις:

1. Model-based (Fictitious Play): Ο πράκτορας διατηρεί belief (πεποίθηση) για τη στρατηγική του αντιπάλου, βασισμένο στην εμπειρική συχνότητα των κινήσεών του, και παίζει Best Response (BR) σε αυτό. Θεωρητικά, η εμπειρική κατανομή συγκλίνει στο Nash σε zero-sum παίγνια.

2. Model-free (Q-Learning): Ο πράκτορας μαθαίνει τις αξίες των ενεργειών (Q-values) μέσω αλληλεπίδρασης, χωρίς να μοντελοποιεί ρητά τον αντίπαλο. Η ενημέρωση βασίζεται στην εξίσωση Bellman· χρησιμοποιείται ε-greedy exploration για την εξισορρόπηση exploration–exploitation.

### 1.4 Μετρικές Απόδοσης και No-Regret

Για την αξιολόγηση χρησιμοποιούμε: (α) απόσταση από το Nash (L2 της τρέχουσας στρατηγικής από το θεωρητικό NE), (β) cumulative reward, (γ) external regret (πόσο κέρδισε ο παίκτης έναντι του καλύτερου σταθερού action in hindsight) και (δ) exploitability (πόσο μπορεί να κερδίσει κάποιος παίζοντας best response απέναντι στη τρέχουσα στρατηγική). Οι αλγόριθμοι no-regret (Hannan consistent) έχουν sublinear external regret και συνδέονται με τη σύγκλιση σε ισορροπία σε repeated games.

---

## 2. Αρχιτεκτονική Συστήματος και Ανάλυση Κώδικα

Η υλοποίηση είναι αρθρωτή σε Python, με διαχωρισμό παιγνίου, πρακτόρων, μετρικών και πειραμάτων.

### 2.1 Δομή και Ρόλος των Αρχείων

games/  
Περιέχει τα παίγνια. Η κλάση BaseGame (base_game.py) ορίζει την αφηρημένη διεπαφή: payoff matrix, get_payoff(action1, action2), get_nash_equilibrium(), best_response(opponent_strategy, player_id). Τα matching_pennies.py και rps.py κληρονομούν από BaseGame και ορίζουν τους πίνακες αποδόσεων και το NE (Matching Pennies: (0.5, 0.5), RPS: (1/3, 1/3, 1/3)). Το grid_game.py υλοποιεί στοχαστικό, turn-based παίγνιο Hunter–Prey σε πλέγμα 3×3: κατάσταση (θέσεις Hunter/Prey), κινήσεις Up/Right/Down/Left/Stay, rewards +10/−10 για capture/timeout και distance-based shaping· δεν κληρονομεί από BaseGame γιατί δεν είναι normal-form.

agents/  
BaseAgent (base_agent.py) ορίζει act(game), update(action, reward, opponent_action), get_strategy(), reset(). Ο FictitiousPlayAgent (fictitious_play.py) διατηρεί opponent_history και belief (εμπειρική κατανομή)· σε κάθε γύρο παίζει game.best_response(belief, player_id) και ενημερώνει το belief αθροιστικά. Ο QLearningAgent (q_learning.py) για matrix games έχει Q-values ανά action, ενημέρωση Q(a) += α(reward − Q(a)), και ε-greedy με optional decay. Ο StochasticQLearningAgent (stochastic_q_learning.py) χρησιμοποιεί Q(s,a) και ενημέρωση Bellman με discount γ· προορίζεται για grid game. Ο MinimaxAgent (minimax.py) κάνει αναζήτηση σε βάθος με evaluation function (π.χ. απόσταση Hunter–Prey) και χρησιμοποιείται μόνο στο grid game ως Hunter.

analysis/  
metrics.py: distance_to_nash(strategy, nash), exploitability(strategy, game, player_id), cumulative_reward(reward_history), external_regret(agent_history, opponent_history, game, player_id) και external_regret_history για time series. visualizer.py: συναρτήσεις για strategy evolution, distance to Nash, cumulative reward, σύγκριση πολλαπλών πρακτόρων, average payoff, exploitability heatmap· αποθήκευση σε results/plots/{game}/{agent_combo}/{plot}.png. gif_maker.py: δημιουργία animations από strategy/distance history.

experiments/  
runner.py: ExperimentRunner διαχειρίζεται matrix games (simultaneous play, get_payoff, get_nash_equilibrium) και turn-based games (step, get_current_player, reset)· σε κάθε iteration καταγράφει strategies, distances, rewards, actions. Τα fp_vs_fp.py, fp_vs_rl.py, rl_vs_rl.py τρέχουν τα αντίστοιχα σεναρία για Matching Pennies και RPS (π.χ. 10.000 iterations), παράγουν plots και γράφουν results.txt. Το minmax_vs_rl_grid.py τρέχει Hunter (Minimax) vs Prey (Stochastic Q-Learning) στο grid με 200.000 turns.

### 2.2 Σύστημα Ανταμοιβών

Στα matrix games (Matching Pennies, RPS) τα payoffs είναι συμμετρικά (+1/−1 ή 0 για ισοπαλία στο RPS), ώστε R₁ + R₂ = 0 και η σύγκριση με το θεωρητικό NE να είναι άμεση. Στο Grid Game: Hunter +10 για capture, Prey −10· timeout −10 Hunter, +10 Prey· επιπλέον distance-based shaping για μη-τελικές κινήσεις. Δεν υπάρχει ρητό κόστος κίνησης· ο παράγοντας έκπτωσης γ=0.99 στο RL δημιουργεί έμμεση πίεση χρόνου.

### 2.3 Παράμετροι Πειραμάτων

Για matrix games: 10.000 iterations (αρκετές για σύγκλιση beliefs και σταθεροποίηση μετρικών), learning rate α=0.1, ε=0.1 (ε-greedy), seed=42 (reproducibility). Για grid: 200.000 turns, Minimax depth=4 (ή 6 αν χρησιμοποιηθεί), γ=0.99, epsilon decay ώστε το Prey να περάσει από exploration σε exploitation. Οι παράμετροι επιλέχθηκαν ώστε να φανούν τα φαινόμενα σύγκλισης και η αστάθεια του RL self-play, χωρίς υπερβολικό tuning.

---

## 3. Πειραματική Ανάλυση και Αποτελέσματα

Τα τρία παίγνια (Matching Pennies, Rock-Paper-Scissors, Grid Hunter-Prey) είναι απλά ως προς τον ορισμό και το μέγεθος του χώρου ενεργειών, αλλά επαρκώς πλούσια ώστε να διακρίνει κανείς τη συμπεριφορά κάθε αλγορίθμου: matrix games με γνωστό θεωρητικό NE για FP και RL, στοχαστικό turn-based παίγνιο για τη σύγκριση Minimax με learning. Δεν επιλέξαμε πολύπλοκα περιβάλλοντα· η έμφαση δόθηκε στη διεξοδική κατανόηση και στην ερμηνεία των αποτελεσμάτων.

### 3.1 Matching Pennies (2×2)

Το θεωρητικό NE είναι (0.5, 0.5). Όλα τα πειράματα τρέχουν με 10.000 iterations, seed=42.

#### Α. FP vs FP

Δύο πράκτορες Fictitious Play παίζουν αντίπαλοι. Παράμετροι: ίδιοι και για τους δύο (belief από εμπειρική συχνότητα, best response).

Αποτελέσματα (από results.txt): FP1 final distance to Nash 0.0007, FP2 0.0092. FP1 cumulative reward 60, FP2 −60. FP1 average reward 0.0060, FP2 −0.0060.

Η εμπειρική συχνότητα (belief) συγκλίνει γρήγορα στο 0.5. Η απόσταση από το Nash τείνει στο μηδέν και για τους δύο (Εικόνα 2). Τα cumulative rewards παραμένουν κοντά στο μηδέν με μικρή ασυμμετρία (60 vs −60), πράγμα που δείχνει ότι και οι δύο πλησιάζουν το mixed NE και κανείς δεν εκμεταλλεύεται συστηματικά τον άλλον.

<img src="results/plots/matching_pennies/FP%20vs%20FP/strategy1.png" width="380" alt="Εξέλιξη στρατηγικής Παίκτης 1" />

Εικόνα 1: Εξέλιξη της πιθανότητας να παίξει «Heads» ο Παίκτης 1 (FP1). Η καμπύλη τείνει στο 0.5.

<img src="results/plots/matching_pennies/FP%20vs%20FP/distance_comparison.png" width="380" alt="Απόσταση από Nash FP vs FP" />

Εικόνα 2: Ευκλείδεια απόσταση της εμπειρικής στρατηγικής από το Nash (0.5, 0.5). Και οι δύο πράκτορες έχουν απόσταση κοντά στο 0.

<img src="results/plots/matching_pennies/FP%20vs%20FP/reward_comparison.png" width="380" alt="Cumulative reward FP vs FP" />

Εικόνα 3: Cumulative reward. Οι καμπύλες παραμένουν κοντά στο μηδέν με μικρή τελική διαφορά (±60).

#### Β. FP vs RL

Εδώ ο Row Player είναι FP και ο Column Player είναι Q-learning (lr=0.1, ε=0.1). Στόχος: να δούμε αν το RL μπορεί να εκμεταλλευτεί την προβλεψιμότητα της FP.

Αποτελέσματα (από results.txt): FP final distance to Nash 0.0000, RL 0.6642. FP cumulative reward −374, RL 374. FP average reward −0.0374, RL 0.0374. FP final external regret 388, RL final external regret −374.

Η FP παραμένει στο Nash (απόσταση 0), αλλά ο RL δεν συγκλίνει στο NE· η απόστασή του παραμένει υψηλή (0.6642). Ο RL μαθαίνει να προβλέπει την determinist αντίδραση της FP (best response στο belief) και την εκμεταλλεύεται· γι’ αυτό το cumulative reward του RL είναι +374 και της FP −374. Το external regret της FP αυξάνεται (388)· η FP δεν είναι no-regret απέναντι σε αυτόν τον δυναμικό αντίπαλο. Το αρνητικό regret του RL (−374) αντιστοιχεί στο ότι κέρδισε περισσότερο από το «best fixed action in hindsight» λόγω της προσαρμοστικής του συμπεριφοράς.

<img src="results/plots/matching_pennies/FP%20vs%20RL/external_regret.png" width="380" alt="External regret FP vs RL" />

Εικόνα 4: Συσσωρευμένο external regret. Η FP (θετική καμπύλη) έχει γραμμικά αυξανόμενο regret· ο RL (αρνητική) έχει sublinear/αρνητικό regret.

<img src="results/plots/matching_pennies/FP%20vs%20RL/avg_payoff.png" width="380" alt="Μέσο payoff FP vs RL" />

Εικόνα 5: Μέσο payoff ανά γύρο. Ο RL έχει θετικό μέσο payoff, η FP αρνητικό.

<img src="results/plots/matching_pennies/FP%20vs%20RL/distance_comparison.png" width="380" alt="Απόσταση από Nash FP vs RL" />

Εικόνα 6: Απόσταση από Nash. FP πλησιάζει 0, RL παραμένει περίπου 0.66.

<img src="results/plots/matching_pennies/FP%20vs%20RL/reward_comparison.png" width="380" alt="Cumulative reward FP vs RL" />

Εικόνα 7: Cumulative reward. RL +374, FP −374.

Συμπέρασμα για FP vs RL (Matching Pennies): Η FP είναι βέλτιστη για εύρεση Nash σε στατικό ή συμμετρικό περιβάλλον, αλλά ευάλωτη σε αντιπάλους που μαθαίνουν τα patterns της· ο RL εκμεταλλεύεται αυτή την προβλεψιμότητα και πετυχαίνει υψηλό cumulative reward εις βάρος της ισορροπίας.

#### Γ. RL vs RL

Δύο Q-learning πράκτορες με τις ίδιες παραμέτρους.

Αποτελέσματα (από results.txt): RL1 και RL2 final distance to Nash 0.6642. RL1 cumulative reward −4, RL2 4. RL1 average reward −0.0004, RL2 0.0004.

Καμία από τις δύο πλευρές δεν συγκλίνει στο Nash (απόσταση ~0.66). Τα cumulative rewards είναι σχεδόν μηδέν (±4), δηλαδή δεν υπάρχει συστηματική εκμετάλλευση· το περιβάλλον είναι non-stationary (ο αντίπαλος αλλάζει συνεχώς), οπότε οι υποθέσεις σύγκλισης του απλού Q-learning δεν ικανοποιούνται. Το heatmap exploitability δείχνει υψηλή exploitability για πολλές τιμές learning rate και epsilon.

<img src="results/plots/matching_pennies/RL%20vs%20RL/exploitability_heatmap.png" width="380" alt="Exploitability heatmap RL vs RL" />

Εικόνα 8: Heatmap exploitability για διάφορες παραμέτρους. Υψηλές τιμές δείχνουν ότι οι learned στρατηγικές απέχουν από το NE.

---

### 3.2 Rock-Paper-Scissors (3×3)

Το θεωρητικό NE είναι (1/3, 1/3, 1/3). Πειράματα με 10.000 iterations.

#### FP vs FP

Αποτελέσματα (από results.txt): FP1 και FP2 final distance to Nash 0.0032. Cumulative rewards 0, 0. Average rewards 0.0000. Και οι δύο πράκτορες συγκλίνουν κοντά στο (1/3, 1/3, 1/3)· τα rewards είναι μηδενικά, συνεπής με την ισορροπία.

#### FP vs RL

Αποτελέσματα (από results.txt): FP final distance to Nash 0.0152, RL 0.7670. FP cumulative reward −213, RL 213. FP final external regret 234, RL −3.

Η FP πλησιάζει το Nash (απόσταση 0.0152)· ο RL παραμένει μακριά (0.7670). Ο RL κερδίζει σε cumulative reward (+213) και έχει σχεδόν μηδενικό external regret (−3)· η FP έχει θετικό και αυξανόμενο regret (234). Η ερμηνεία είναι ίδια με το Matching Pennies: ο RL δεν στοχεύει στο Nash (payoff 0), αλλά παίζει best response στις μικρές στατιστικές αποκλίσεις της FP και μεγιστοποιεί το κέρδος του.

<img src="results/plots/rock_paper_scissors/FP%20vs%20RL/distance_comparison.png" width="380" alt="Απόσταση Nash RPS FP vs RL" />

Εικόνα 9: Απόσταση από Nash στο RPS. FP (μπλε) κοντά στο 0, RL (πορτοκαλί) ~0.77.

<img src="results/plots/rock_paper_scissors/FP%20vs%20RL/reward_comparison.png" width="380" alt="Cumulative reward RPS FP vs RL" />

Εικόνα 10: Cumulative reward RPS. RL +213, FP −213.

#### RL vs RL

Αποτελέσματα (από results.txt): RL1 και RL2 final distance to Nash 0.7670. RL1 cumulative 212, RL2 −212. Average rewards 0.0212 και −0.0212.

Μεγάλες αποστάσεις από Nash και ισχυρά θετικά/αρνητικά cumulative rewards δείχνουν κυκλική ή εκμεταλλευτική δυναμική· οι πολιτικές δεν σταθεροποιούνται στο (1/3, 1/3, 1/3).

---

### 3.3 Grid Game (Hunter–Prey, στοχαστικό, turn-based)

Συγκρίνουμε Hunter με Minimax (βάθος 4) και Prey με Stochastic Q-Learning. Πλέγμα 3×3, max_steps=20, 200.000 turns. Δύο σεναρία: Hunter first και Prey first.

Hunter first (από results.txt): Iterations 200.000, Episodes 10.264, Hunter turns 100.335, Captures 670, Capture rate 6.5%. Hunter total reward 6998.50, mean reward 0.0698. Prey final epsilon 0.068.

Ο Minimax Hunter κυριαρχεί: θετική συσσωρευμένη ανταμοιβή και capture rate 6.5%. Το RL Prey ξεκινά με μεγάλες απώλειες αλλά το epsilon μειώνεται (0.068) και μαθαίνει να αποφεύγει· η καμπύλη μέσου reward «ισιώνει» ελαφρώς με το χρόνο.

Prey first: Capture rate 6.6%, Hunter total reward αρνητικό (−87868.80) λόγω timeout penalties (το Prey κινείται πρώτο και διαφεύγει συχνά)· το RL Prey ωφελείται από τη σειρά κίνησης.

<img src="results/plots/grid_game/MinMax%20vs%20RL%20-%20Hunter%20first/cumulative_reward.png" width="380" alt="Cumulative reward Grid Hunter first" />

Εικόνα 11: Συσσωρευμένο reward στο Grid Game (Hunter first). Hunter (Minimax) θετική τάση.

<img src="results/plots/grid_game/MinMax%20vs%20RL%20-%20Hunter%20first/avg_reward.png" width="380" alt="Μέσο reward Grid" />

Εικόνα 12: Μέσο reward (κινητός μέσος). Ο Hunter έχει θετικό μέσο reward· το Prey βελτιώνεται με το χρόνο.

Συμπέρασμα Grid: Όταν το μοντέλο του κόσμου είναι γνωστό, η αναζήτηση (Minimax) δίνει άμεσο πλεονέκτημα· το RL Prey απαιτεί πολλά επεισόδια για να χαρτογραφήσει το state space και να μειώσει τις απώλειες.

---

## 4. Συμπεράσματα από τα Πειράματα

Σύγκλιση στο Nash και συμπεριφορά FP: Σε FP vs FP (Matching Pennies και RPS) οι τελικές αποστάσεις από το Nash είναι πολύ μικρές (0.0007, 0.0092 στο MP· 0.0032 και στα δύο στο RPS) και τα cumulative rewards κοντά στο μηδέν. Αυτό επιβεβαιώνει ότι η Fictitious Play συγκλίνει προς το mixed Nash σε zero-sum matrix games όταν και οι δύο παίκτες τη χρησιμοποιούν.

FP Exploitment από RL: Σε FP vs RL, στα δύο παίγνια η FP παραμένει κοντά στο Nash (απόσταση 0 ή 0.0152) αλλά ο RL έχει μεγάλη απόσταση (0.6642, 0.7670) και σημαντικά μεγαλύτερο cumulative reward (MP: +374 vs −374· RPS: +213 vs −213). Το RL δεν συγκλίνει στο NE αλλά μαθαίνει να παίζει best response στις εμπειρικές συχνότητες της FP, οπότε την εκμεταλλεύεται. Αυτό φαίνεται και από το external regret: η FP έχει υψηλό θετικό regret (388, 234), ενώ ο RL έχει αρνητικό ή σχεδόν μηδενικό (−374, −3)· δηλαδή η FP δεν είναι no-regret απέναντι σε αυτόν τον αντίπαλο.

RL vs RL και αστάθεια: Στο RL vs RL και στα δύο παίγνια οι αποστάσεις από το Nash παραμένουν υψηλές (0.6642, 0.7670) και τα cumulative rewards είτε σχεδόν μηδέν (MP: −4, +4) είτε μεγάλα και αντίθετα (RPS: 212, −212). Το περιβάλλον είναι non-stationary και το απλό Q-learning δεν συγκλίνει σε σταθερή ισορροπία· τα αποτελέσματα είναι συνεπή με κυκλική ή ασταθή δυναμική και με το ότι η exploitability παραμένει υψηλή για πολλές παραμέτρους.

No-regret και απόδοση: Η ελαχιστοποίηση του external regret (no-regret / Hannan consistency) συνδέεται με τη θεωρία repeated games. Στα πειράματά μας, όταν ο RL αντιμετωπίζει FP, έχει αρνητικό ή πολύ μικρό regret και υψηλό cumulative reward· η FP έχει γραμμικά αυξανόμενο regret. Η απόδοση «σε πραγματικό χρόνο» (cumulative reward) ευνοεί τον πράκτορα που προσαρμόζεται (RL) έναντι του που συγκλίνει στο Nash αλλά μένει προβλέψιμος (FP).

Search vs Learning στο Grid: Στο Grid Game, ο Hunter με Minimax (βάθος 4) πετυχαίνει capture rate 6.5% και θετικό συνολικό reward (6998.5) όταν κινείται πρώτος. Το RL Prey μειώνει το epsilon με το χρόνο και δείχνει βελτίωση, αλλά η εκ των προτέρων γνώση του μοντέλου (Minimax) δίνει άμεσο πλεονέκτημα· το RL χρειάζεται μεγάλο αριθμό επεισοδίων για να μάθει το state space.

Συνολικά, η μελέτη δείχνει ξεκάθαρα τη διαφορά μεταξύ αλγορίθμων που στοχεύουν στη σύγκλιση στο Nash (FP) και αλγορίθμων που μεγιστοποιούν reward απέναντι σε προσαρμοστικούς αντιπάλους (RL)· τα αριθμητικά αποτελέσματα (αποστάσεις, cumulative rewards, external regret) και τα plots τεκμηριώνουν αυτά τα συμπεράσματα. Η εργασία παραμένει στο πλαίσιο απλών προβλημάτων και απλών παραδειγμάτων, ώστε η βαθιά κατανόηση και η ξεκάθαρη απεικόνιση δυνατοτήτων και αδυναμιών των αλγορίθμων να μην θυσιάζονται σε περιττή πολυπλοκότητα.
