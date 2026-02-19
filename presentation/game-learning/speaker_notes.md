# Σημειώσεις ομιλίας — 15 διαφάνειες

## 1. Τίτλος
Καλημέρα. Παρουσίαση: Fictitious Play και Reinforcement Learning για υπολογισμό ισορροπιών σε zero-sum και στοχαστικά παίγνια.

## 2. Περίληψη
Μελέτη συμπεριφοράς πρακτόρων σε ανταγωνιστικά περιβάλλοντα. Σύγκριση FP και Q-learning. Πειραματική αξιολόγηση σε τρία παίγνια: Matching Pennies, Rock-Paper-Scissors, Grid Hunter-Prey. Κεντρικό εύρημα: η προσέγγιση του Nash δεν ταυτίζεται απαραίτητα με το καλύτερο reward όταν ο αντίπαλος μαθαίνει.

## 3. Θεωρητικό Πλαίσιο — Πράκτορες
Ορθολογικοί πράκτορες (MEU). Σε περιβάλλοντα multi-agent η χρησιμότητα εξαρτάται και από τις ενέργειες των άλλων. Στρατηγική σκέψη: Best Response, Mixed Strategies, Equilibrium.

## 4. Zero-Sum και Nash Equilibrium
Zero-sum: U₁+U₂=0. Nash: αμοιβαία best responses. MP Nash (0.5, 0.5), RPS (1/3, 1/3, 1/3). Θεώρημα Minimax: max min = min max.

## 5. Fictitious Play (FP)
Model-based: belief από εμπειρική συχνότητα, Best Response σε κάθε γύρο. Συγκλίνει στο Nash σε zero-sum, αλλά είναι ντετερμινιστική και ευάλωτη σε προσαρμοστικούς αντιπάλους.

## 6. Q-Learning
Model-free: Q-values, Bellman update, ε-greedy. Μαθαίνει χωρίς μοντέλο αντιπάλου. Σε non-stationary (RL vs RL) δεν συγκλίνει σε Nash.

## 7. Αρχιτεκτονική & Μετρικές
Αρχιτεκτονική: games, agents, experiments. Μετρικές: Distance to Nash, Cumulative Reward, External Regret, Exploitability.

## 8. Matching Pennies — Αποτελέσματα
Πίνακας: FP vs FP σύγκλιση, FP vs RL εκμετάλλευση (+374 RL), RL vs RL αστάθεια. Κρίσιμη παρατήρηση: ο RL κερδίζει χωρίς να είναι κοντά στο Nash.

## 9. Rock-Paper-Scissors — Αποτελέσματα
FP vs FP: τέλεια σύγκλιση (0, 0). FP vs RL: RL +213. RL vs RL: κυκλική δυναμική, ασυμμετρία +212/−212.

## 10. External Regret & Exploitability
FP γραμμικά αυξανόμενο regret vs RL. Exploitability εξαρτάται από ε. Κρίσιμο συμπέρασμα: εγγύτητα στο Nash ≠ ανωτερότητα σε reward απέναντι σε adaptive αντίπαλο.

## 11. Grid Hunter-Prey Setup
Πλέγμα 3×3, turn-based, 20 steps. Hunter (Minimax ή RL) vs Prey (RL). Rewards: Capture +10/−10, Timeout −10/+10. 200k turns.

## 12. Grid — MinMax vs RL
Hunter first: +6,998, capture 6.5%. Prey first: −87,868 λόγω timeouts. Η σειρά κίνησης αλλάζει δραματικά το reward παρά παρόμοιο capture rate.

## 13. Grid — RL vs RL
Hunter κυριαρχεί: 88.9% και 85.6% capture (έναντι 6.5% MinMax vs RL). Συμπρωτεύουσα εξέλιξη υπέρ του Hunter — μαθαίνει να κυνηγά πιο αποτελεσματικά.

## 14. Κεντρικά Συμπεράσματα
(1) FP vs FP: ισορροπία. (2) FP vs RL: εκμετάλλευση. (3) RL vs RL: αστάθεια. (4) Grid: υπεροχή RL. (5) Σημασία σειράς κίνησης. Βασικό: Nash ≠ καλύτερο reward με adaptive αντίπαλο.

## 15. Τελικές Σκέψεις
FP για equilibrium computation. Minimax για robust ασφάλεια. Q-Learning για πρακτική απόδοση. Οδηγός επιλογής με βάση το ζητούμενο. Ευχαριστώ.
