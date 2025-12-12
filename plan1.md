# Πλάνο Εργασίας: Fictitious Play & RL for Computing Equilibria

**Μάθημα:** Intelligent Agents and Multiagent Systems  
**Θέμα:** D. Fictitious Play and Reinforcement Learning for computing equilibria  
**Ομάδα:** [Ονόματα - 2 άτομα]  
**Κατανομή:** 30% θεωρία, 70% υλοποίηση

---

## 📅 Χρονοδιάγραμμα

- **20 Δεκεμβρίου:** Δήλωση θέματος
- **27-31 Ιανουαρίου:** Παρουσίαση 1ης έκδοσης (feedback)
- **15 Φεβρουαρίου:** Τελική παρουσίαση

---

## 🎯 Στόχοι Εργασίας

1. Κατανόηση repeated & zero-sum stochastic games
2. Υλοποίηση Fictitious Play algorithm
3. Υλοποίηση Reinforcement Learning algorithms (Q-learning, Nash Q-learning)
4. Πειραματική σύγκριση FP vs RL σε διάφορα game scenarios
5. Ανάλυση trade-offs: convergence speed, solution quality, robustness

---

## 📚 Θεωρητικό Μέρος (30%)

### 1. Game Theory Foundations
- **Repeated games:** finite vs infinite horizon
- **Zero-sum games:** minimax theorem, mixed strategies
- **Stochastic games (Markov games):** states, transitions, policies
- **Nash equilibrium:** pure vs mixed, existence theorems

### 2. Fictitious Play (FP)
- **Αλγόριθμος:** belief updates, best response computation
- **Θεωρία σύγκλισης:** πότε converge (2-player zero-sum), πότε όχι
- **Πλεονεκτήματα/Μειονεκτήματα:** simple, theoretically grounded, αλλά αργό

### 3. Reinforcement Learning (RL)
- **Q-Learning:** state-action values, Bellman equation, ε-greedy exploration
- **Nash Q-Learning:** multiagent extension, computing Nash on Q-values
- **Policy Gradient methods** (optional/advanced)
- **Convergence properties:** independent learners vs coordinated learning

### 4. Equilibrium Concepts
- Nash equilibrium
- ε-Nash equilibrium (approximate)
- Exploitability measure

---

## 💻 Υλοποίηση (70%)

### Phase 1: Environment Setup
**Υλοποίηση Game Environments:**
- **Matching Pennies** (2x2 zero-sum)
- **Rock-Paper-Scissors** (3x3 no pure Nash)
- **Prisoner's Dilemma** (repeated, cooperation dynamics)
- **Grid Soccer** (stochastic game, 2-agent soccer)

**Deliverable:** Game classes με:
- State representation
- Action spaces
- Reward functions
- Transition dynamics (για stochastic games)

---

### Phase 2: Fictitious Play Implementation
**Core Algorithm:**
- Belief tracking για opponent's strategy
- Best response computation
- Strategy update mechanism

**Features:**
- Memory of opponent's action history
- Empirical frequency distribution
- Convergence monitoring (distance to Nash)

**Deliverable:** FP agent class που παίζει οποιοδήποτε game

---

### Phase 3: Reinforcement Learning Implementation
**Algorithm 1: Q-Learning**
- Q-table initialization
- ε-greedy exploration
- Q-value updates (learning rate α, discount γ)
- Decay schedules για ε και α

**Algorithm 2: Nash Q-Learning** (για stochastic games)
- Joint action Q-values
- Nash equilibrium solver για stage game
- Coordination mechanism

**Deliverable:** RL agent classes (Q-learning, Nash Q-learning)

---

### Phase 4: Experimental Framework
**Metrics:**
- **Convergence speed:** rounds to reach ε-Nash
- **Solution quality:** exploitability, distance from theoretical Nash
- **Robustness:** adaptation σε payoff changes
- **Computational cost:** time per iteration, memory usage

**Experiments:**

1. **Experiment 1: Simple Zero-Sum Games**
   - Games: Matching Pennies, RPS
   - Compare: FP vs Q-Learning
   - Measure: convergence rounds, final strategy

2. **Experiment 2: Repeated Games**
   - Game: Iterated Prisoner's Dilemma
   - Compare: FP vs Q-Learning
   - Measure: cooperation emergence, payoff over time

3. **Experiment 3: Stochastic Games**
   - Game: Grid Soccer
   - Compare: FP vs Nash Q-Learning
   - Measure: win rate, state-action distribution

4. **Experiment 4: Robustness Test**
   - Change payoffs mid-game
   - Measure: adaptation speed

5. **Experiment 5: Scaling**
   - Vary state space size / action space
   - Measure: performance degradation

**Deliverable:** 
- Experiment scripts
- Data collection pipeline
- Statistical analysis (mean, std, confidence intervals)

---

## 📊 Παραδοτέα

### 1. Report (Αναφορά)
**Δομή:**
- **Introduction:** Problem definition, motivation
- **Theoretical Background:**
  - Game theory basics
  - FP algorithm & theory
  - RL algorithms & theory
- **Implementation:** Architecture, design choices, pseudocode
- **Experimental Results:**
  - Setup description
  - Results per experiment (plots, tables)
  - Statistical analysis
- **Discussion:** 
  - FP vs RL comparison
  - When to use each
  - Limitations & future work
- **Conclusion**
- **References**

**Εκτίμηση:** 15-25 σελίδες

---

### 2. Presentation (Παρουσίαση)
**Διάρκεια:** ~20-30 λεπτά + Q&A

**Δομή:**
- Introduction (2 slides): Problem & Goals
- Theory (4-5 slides): 
  - Games overview
  - FP algorithm με example
  - RL algorithm με example
- Implementation (3-4 slides): Architecture, key design choices
- Results (6-8 slides):
  - Experiment 1 results με plots
  - Experiment 2 results με plots
  - Comparative analysis
- Discussion (2-3 slides): Key findings, trade-offs
- Conclusion (1 slide)

**Εκτίμηση:** 18-25 slides

---

### 3. Code
**Δομή repository:**
```
project/
├── README.md              # Setup instructions, how to run
├── requirements.txt       # Dependencies
├── src/
│   ├── games/            # Game environment implementations
│   ├── agents/           # FP and RL agent classes
│   ├── utils/            # Helper functions
│   └── experiments/      # Experiment scripts
├── results/              # Experimental data, plots
├── notebooks/            # Jupyter notebooks για analysis
└── docs/                 # Documentation
```

**Requirements:**
- Clean, documented code
- README με οδηγίες εκτέλεσης
- Reproducible experiments (fixed seeds)

---

## 🔧 Tools & Technologies

**Προτεινόμενα:**
- **Language:** Python 3.8+
- **Libraries:**
  - NumPy (matrices, computations)
  - Matplotlib/Seaborn (plots)
  - Pandas (data handling)
  - OpenAI Gym (optional, για standard interface)
  - nashpy (για Nash equilibrium computation)
- **Version Control:** Git/GitHub
- **Documentation:** Markdown, Jupyter notebooks

---

## 👥 Κατανομή Εργασίας (Πρόταση)

### Άτομο 1:
- Θεωρητικό μέρος: FP, repeated games
- Implementation: FP algorithm, simple games
- Experiments 1, 2, 4
- Report: Theory sections, FP parts

### Άτομο 2:
- Θεωρητικό μέρος: RL, stochastic games
- Implementation: RL algorithms, stochastic games
- Experiments 3, 5
- Report: RL sections, results analysis

### Κοινή Δουλειά:
- Experimental framework design
- Results analysis & plots
- Presentation preparation
- Code review & integration

---

## 📖 Βασική Βιβλιογραφία

1. **Shoham & Leyton-Brown** - "Multiagent Systems" (κύριο βιβλίο)
   - Chapters: Game Theory, Learning in games
2. **Sutton & Barto** - "Reinforcement Learning: An Introduction"
   - Chapters: Q-learning, Multi-agent RL
3. **Papers:**
   - Fictitious Play original (Brown 1951)
   - Nash Q-Learning (Hu & Wellman 2003)
   - Multiagent RL survey papers

---

## ✅ Milestones

**Εβδομάδα 1-2 (μέχρι 5 Ιαν):**
- Βιβλιογραφική επισκόπηση
- Θεωρητική μελέτη FP & RL
- Setup environment

**Εβδομάδα 3 (6-12 Ιαν):**
- Implementation: Simple games
- Implementation: FP algorithm
- Αρχικά tests

**Εβδομάδα 4 (13-19 Ιαν):**
- Implementation: RL algorithms
- Implementation: Stochastic games
- Integration testing

**Εβδομάδα 5 (20-26 Ιαν):**
- Run experiments 1, 2, 3
- Preliminary results analysis
- **Προετοιμασία 1ης παρουσίασης**

**27-31 Ιαν: 1η Παρουσίαση & Feedback**

**Εβδομάδα 6 (27 Ιαν - 2 Φεβ):**
- Ενσωμάτωση feedback
- Experiments 4, 5
- Advanced analysis

**Εβδομάδα 7 (3-9 Φεβ):**
- Report writing
- Plots & visualizations
- Code cleanup & documentation

**Εβδομάδα 8 (10-15 Φεβ):**
- Τελική παρουσίαση
- Final review
- **Παράδοση**

---

## 🎓 Success Criteria

- ✅ Σωστή θεωρητική κατανόηση FP & RL
- ✅ Working implementations που converge
- ✅ Comprehensive experiments με clear results
- ✅ Insightful comparison & analysis
- ✅ High-quality presentation & report
- ✅ Clean, documented code

---

## 💡 Extra Ideas (αν έχουμε χρόνο)

- Deep Q-Networks για μεγαλύτερα state spaces
- Opponent modeling techniques
- Multi-agent scenarios (N>2 agents)
- Real-world application demo
- Interactive visualization tool

---

**Ερωτήσεις/Σχόλια:** [email/chat]

Let's build something great! 🚀
