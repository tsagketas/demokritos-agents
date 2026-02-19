# Σημειώσεις ομιλίας — 15 διαφάνειες

## 1. Τίτλος
Καλημέρα. Παρουσίαση: Fictitious Play και Reinforcement Learning για υπολογισμό ισορροπιών σε zero-sum και στοχαστικά παίγνια.

## 2. Θεωρητικό πλαίσιο
Zero-sum: U₁+U₂=0. Nash equilibrium, best response. MP: (0.5,0.5), RPS: (1/3,1/3,1/3). Θεώρημα Minimax: max min = min max.

## 3. FP και Q-Learning
FP: belief από εμπειρική συχνότητα, Best Response. Συγκλίνει στο Nash αλλά ευάλωτη. Q-Learning: Q-values, ε-greedy, model-free. Προσαρμόζεται αλλά σε RL vs RL δεν συγκλίνει σε Nash.

## 4. Μετρικές και παιγνία
Απόσταση Nash, exploitability, cumulative reward, external regret. Τρία παίγνια: MP, RPS, Grid. Παράμετροι: 10k iterations matrix, 200k grid.

## 5. Σύνοψη αποτελεσμάτων
Πλήρης πίνακας: όλα τα σεναρία, αποστάσεις, rewards, παρατηρήσεις.

## 6. Ροή πειραμάτων
Matrix: act ταυτόχρονα, payoffs, update, metrics. Grid: turn-based, step, update με next_state. Αρχιτεκτονική: games, agents, experiments, analysis.

## 7. MP FP vs FP
Αποτελέσματα: distances 0.0007, 0.0092. Cumulative ±60. Διαγράμματα: strategy, distance, reward.

## 8. MP FP vs RL
FP στο Nash, RL +374. Regret FP 388. Τέσσερα διαγράμματα: external regret, avg payoff, distance, reward.

## 9. MP RL vs RL
Distance 0.66 και στα δύο. Πίνακας exploitability α/ε. Heatmap, distance, reward.

## 10. RPS
Πίνακας τριών σεναρίων. FP vs RL, RL vs RL. Διαγράμματα distance, reward, heatmap.

## 11. Grid MinMax vs RL
Πίνακας Hunter/Prey first. Prey first: −87k λόγω timeouts. Τρία διαγράμματα cumulative/avg reward.

## 12. Grid RL vs RL
85–89% capture. RL Hunter κυριαρχεί. Συμπέρασμα: συμπρωτεύουσα εξέλιξη υπέρ Hunter.

## 13. Συμπεράσματα
FP συγκλίνει αλλά ευάλωτη. RL εκμεταλλεύεται αλλά δεν συγκλίνει σε Nash matrix. Grid: RL ισχυρό. Επιλογή ανάλογα με ζητούμενο.

## 14. Exploitability και No-Regret
Exploitability εξαρτάται από ε. No-regret: FP δεν είναι. Σύστημα ανταμοιβών.

## 15. Τέλος
Ευχαριστώ.
